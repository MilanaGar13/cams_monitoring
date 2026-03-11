mod ping_stats;
mod rtsp;

use axum::{http::HeaderMap, routing::get, Router, body::Bytes};
use axum::extract::Query;
use axum::response::IntoResponse;
use std::sync::{Arc};
use tokio::sync::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::env;
use single_instance::SingleInstance;
use serde::Deserialize;
use tokio::fs;
use std::process;
use tokio::sync::mpsc;
use ffmpeg_next as ffmpeg;
use std::time::{Duration, Instant};
use ffmpeg_next::codec::Parameters;
use std::io::Cursor;
use tch::{CModule, Tensor, Kind};
use image::{DynamicImage, imageops::resize, imageops::FilterType};
use image::{RgbImage, ImageBuffer};
use std::fs::File;
use std::io::Write;
use rand::Rng;
use std::path::Path;


//модели из файлов загружаются в константы байтовые
static MODEL_FROZEN: &[u8] = include_bytes!("../frozen_train/freeze_detector_scripted.pt");
static MODEL_ANOMALY: &[u8] = include_bytes!("../anomaly_train/siamese_model_cpu.pt");
static MODEL_RAKURS: &[u8] = include_bytes!("../rakurs_train/view_model.pt");

/* структура состоит из подструктур */
#[derive(Debug, Deserialize, Clone)]
struct Config {
    global: Global,
    /* структура с двумя переменными текстовыми - где биндить и по какому пути размещать метрики */
    metrics: Metrics,
    /* Vec(массив) со структурами cam */ 
    cams: Vec<Cams>
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
struct Metrics {
    bind: String,
    uri: String
}
/* global - полсто с одним текстовым параметром */
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
struct Global {
    etalons_dir: String
}


/* тут заданы значения по умолчению для конфигов */
impl Default for Global {
    fn default() -> Self {
        Self {
            etalons_dir: "etalons".into()
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            bind: "[::]:9992".into(),
            uri: "/metrics".into()
        }
    }
}

impl Default for Cams {
    fn default() -> Self {
        Self {
            ip: "127.0.0.1".into(),
            rtsp_port: 554,
            rtsp_uri: "/".into(),
            password: "".into(),
            username: "".into(),
            rtsp_transport: "tcp".into()
        }
    }
}
#[derive(Clone)]
struct StreamStats {
    host: String,
    bytes: u64,
    last_report: Instant,
    params: Parameters,
    media_type: ffmpeg::media::Type,
    rate: f64,
    id: usize
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
struct Cams {
    ip: String,
    rtsp_port: u16,
    rtsp_uri: String,
    username: String,
    password: String,
    rtsp_transport: String
}

#[derive(Clone, Debug)]
struct StreamRgb {
    host: String,
    id: usize,
    rgb: Vec<u8>,
    width: u32,
    height: u32
}

type Metric = HashMap<String, String>;
type ArcMetric = Arc<Mutex<Metric>>;

fn frame_to_tensor(buf: &[u8], width: i64, height: i64) -> Tensor {
    let arr = Tensor::from_slice(buf)
        .reshape(&[height, width, 3])
        .to_kind(Kind::Uint8);
    arr.permute(&[2, 0, 1]) / 255.0
}

fn normalize(v: &Tensor) -> Tensor {
    let norm = (v * v).sum(Kind::Float).sqrt();
    v / norm
}

#[tokio::main]
async fn main() {
    /* Структура для вывода метрик в веб */
    let metrics: ArcMetric =  Arc::new(Mutex::new(HashMap::new()));
    let current_images : Arc<Mutex<HashMap<String, StreamRgb>>> = Arc::new(Mutex::new(HashMap::new()));
    let etalons_images : Arc<Mutex<HashMap<String, StreamRgb>>> = Arc::new(Mutex::new(HashMap::new()));

    /* тут просто код чтобы нельзя было запустить два экземпляра демона */
    let instance_id = format!("cams_monitoring");
    let instance = SingleInstance::new(&instance_id).unwrap();
    if !instance.is_single() {
        eprintln!("Программа уже запущена");
        std::process::exit(1);
    }
    let start_ = Instant::now();

    /* в параметрах запуска указывается файл конфига, либо используется по умолчанию config.yaml, конфик в формате  YAML*/
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("config.yaml");

    /* читаем содержимое файла конфига */
    let contents = match fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => {
            /* ошибка чтения тут выводится */
            eprintln!("Ошибка чтения файла '{}': {}", path, e);
            "".to_string()
        }
    };
    /* если пустой файл или файла нет - выходим */
    if contents == "" {
        process::exit(1);
    }
    /* парсим конфик с помощью serde_yaml и сохраняем в структуру Config */
    let config: Config = match serde_yaml::from_str(&contents) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Ошибка парсинга config файла'{}': {}", path, e);
            process::exit(1);
        }
    };

    /* создаем папку для эталонов, если её еще нет */
    let etalons_dir = config.global.etalons_dir;
    let etalons_dir_ = Path::new(etalons_dir.as_str());

    match fs::create_dir_all(etalons_dir_).await {
        Ok(_ok) => {},
        Err(_e) => {
            println!("Ошибка создания папки с эталонами.");
            process::exit(1);
        }
    }

    let cams = config.cams.clone();

    /* запускаем все потоки для каждой камеры */
    for (_i, cam) in cams.iter().enumerate()  {
        { //pings
            /* поток для пинга камеры и сбора метрик по пингу - rtt, тайминг, потери пакетов */
            tokio::spawn({
                /* перед запуском потока клонируем в него переменные с метриками и с камерой */
                let cam = cam.clone();
                let metrics = metrics.clone();
                async move {
                    /* запускаем функцию из модуля пинг */
                    ping_stats::ping(cam.ip.as_str(), metrics.clone()).await;
                }
            });
        }
        

        //каналы для передачи данных из потоков
        let (tx_stats, mut rx_stats) = mpsc::channel::<StreamStats>(100); //для битрейта
        let (tx_rgb, mut rx_rgb) = mpsc::channel::<StreamRgb>(100); //для аналитики(картинка)
        {//async update rgb - поток для обработки аналитики
            tokio::spawn({
                let metrics = metrics.clone();
                let start_ = start_.clone();
                let current_images = current_images.clone();
                let etalons_dir = etalons_dir.clone();
                let etalons_images = etalons_images.clone();
                let buf_prev_: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::new(Mutex::new(HashMap::new()));
                let firsts_: Arc<Mutex<HashMap<String, bool>>> = Arc::new(Mutex::new(HashMap::new()));
                async move {
                    while let Some(update) = rx_rgb.recv().await {
                        {
                            tokio::spawn({ //каждый кадр, который пришел запускаем в отдельном потоке
                                //модели из констант загружаются в переменные
                                let mut cursor_frozen = Cursor::new(MODEL_FROZEN);
                                let model_frozen = match CModule::load_data(&mut cursor_frozen) {
                                    Ok(ok) => {
                                        ok
                                    },
                                    Err(_e) => {
                                        println!("LOAD FROZEN ERROR: {}", _e);
                                        process::exit(1);
                                    }
                                };
                                let mut cursor_anomaly = Cursor::new(MODEL_ANOMALY);
                                let model_anomaly = match CModule::load_data(&mut cursor_anomaly) {
                                    Ok(ok) => {
                                        ok
                                    },
                                    Err(_e) => {
                                        println!("LOAD ANOMALY ERROR: {}", _e);
                                        process::exit(1);
                                    }
                                };
                                let mut cursor_rakurs = Cursor::new(MODEL_RAKURS);
                                let model_rakurs = match CModule::load_data(&mut cursor_rakurs) {
                                    Ok(ok) => {
                                        ok
                                    },
                                    Err(_e) => {
                                        println!("LOAD RAKURS ERROR: {}", _e);
                                        process::exit(1);
                                    }
                                };

                                let metrics = metrics.clone();
                                let update = update.clone();
                                let buf_prev_ = buf_prev_.clone();
                                let firsts_ = firsts_.clone();
                                let etalons_dir = etalons_dir.clone();
                                let current_images = current_images.clone();
                                let etalons_images = etalons_images.clone();
                                let start_ = start_.clone();
                                async move {
                                    //проверка зависшей картинки
                                    let mut _buf_curr: Vec<u8> = Vec::new();
                                    let mut buf_prev_n: Vec<u8> = Vec::new();
                                    {
                                        let mut _ffirst: bool = true;
                                        {
                                            let mut _first = firsts_.lock().await;
                                            let first_h = _first.entry(update.host.clone()).or_insert(true);
                                            _ffirst = *first_h;
                                        }
                                        {
                                            let mut current_images_ = current_images.lock().await;
                                            let ci = current_images_.entry(update.host.clone()).or_insert(
                                                update.clone()
                                            );
                                            *ci = update.clone();
                                        }
                                        if _ffirst { //если фото для камеры первое -- просто сохраняем в массив и никак не обрабатываем
                                            {
                                                let mut _first = firsts_.lock().await;
                                                let first_h = _first.entry(update.host.clone()).or_insert(false);
                                                *first_h = false;
                                            }
                                            {
                                                let mut buf_prev = buf_prev_.lock().await;
                                                let buf_prev_h = buf_prev.entry(update.host.clone()).or_insert(Vec::new());
                                                buf_prev_h.clone_from(&update.rgb.to_vec());
                                            }
                                        } else { //если уже не первое
                                                _buf_curr = update.rgb.clone().to_vec();
                                                {
                                                    let mut buf_prev = buf_prev_.lock().await;
                                                    let buf_prev_h = buf_prev.entry(update.host.clone()).or_insert(Vec::new());
                                                    buf_prev_n.clone_from(&buf_prev_h);

                                                }
                                                if start_.elapsed() >= Duration::from_secs(240) {
                                                    // _buf_curr.clone_from(&buf_prev_n); //имитируем зависшую картинку, для проверки оповещений
                                                }
                                                //загружаем текущую картинку в тензор
                                                let t1 = frame_to_tensor(&_buf_curr.clone(), update.width as i64, update.height as i64);
                                                //_buf_curr = update.rgb.clone().to_vec();
                                                //загружаем предыдущую картинку в тензор
                                                let t2 = frame_to_tensor(&buf_prev_n.clone(), update.width as i64, update.height as i64);
                                                //делаем срез между ними
                                                let input = Tensor::cat(&[t1.unsqueeze(0), t2.unsqueeze(0)], 1);
                                                //отправляем в модель
                                                match model_frozen.forward_ts(&[input]) {
                                                    Ok(output) => {
                                                        //на выходе модель выдает число, если оно >0.5 = зависла картинка, то значит картинка зависла, но цифру будем обрабатывать уже при уведомлениях
                                                        let prob = output.sigmoid().double_value(&[0, 0]);
                                                        println!("Значение на выходе модели проверки зависшей картинки: {}", prob);
                                                        {//просто как обычно записываем значение в метрики
                                                            let mut m = metrics.lock().await;
                                                            let val_ = format!("{:.5}", prob.to_string());
                                                            let key = format!("frozen{{host=\"{}\", Stream=\"{}\"}}", update.host, update.id);
                                                            let m_ = m.entry(key).or_insert(val_.clone());
                                                            *m_ = val_;
                                                        }
                                                    },
                                                    Err(_e) => {
                                                        println!("ERROR FROZEN: {}", _e);
                                                    }
                                                }
                                                {  //тут текущую картинку записываем в предыдущую, чтобы она была предыдущей для следующей проверки
                                                    let mut buf_prev = buf_prev_.lock().await;
                                                    let buf_prev_h = buf_prev.entry(update.host.clone()).or_insert(Vec::new());
                                                    buf_prev_h.clone_from(&_buf_curr);
                                                }
                                            }
                                    }
                                    //проверка анамалий и ракурса
                                    {
                                        _buf_curr = update.rgb.clone().to_vec();
                                        //предварительно уменьшаем картинку до 320*240, иначе нужны большие ресурсы для обучения модели, обучались на 320*240
                                        let img_buffer: RgbImage = ImageBuffer::from_raw(update.width as u32, update.height as u32, _buf_curr.clone()).expect("Неверный размер буфера");
                                        let img = DynamicImage::ImageRgb8(img_buffer);
                                        let img_resized_curr = resize(&img, 320, 240, FilterType::Triangle);
                                        let resized_rgb_curr: RgbImage = DynamicImage::ImageRgba8(img_resized_curr.clone()).to_rgb8();
                                        let buf_curr_resized: Vec<u8> = resized_rgb_curr.clone().into_raw();

                                        let etalon_file = format!("{}/{}", etalons_dir, update.host.clone());
                                        let etalon_file_raw = format!("{}/{}_raw", etalons_dir, update.host.clone());
                                        let etalon_file_ = Path::new(&etalon_file);
                                        if !etalon_file_.is_file() { //проверяем есть ли "эталон", если нету - записываем в файл, если нужно будет обновить эталон - просто удаляем файл из папки эталонов
                                            match File::create(etalon_file) {
                                                Ok(mut _f) => {
                                                    match _f.write_all(&buf_curr_resized.clone()) {
                                                        Ok(_) => {},
                                                        Err(_) => {}
                                                    }
                                                },
                                                Err(_) => {}
                                            }
                                            match File::create(etalon_file_raw) {
                                                Ok(mut _f) => {
                                                    match _f.write_all(&_buf_curr.clone()) {
                                                        Ok(_) => {},
                                                        Err(_) => {}
                                                    }
                                                },
                                                Err(_) => {}
                                            }
                                        } else {
                                            match fs::read(etalon_file_raw).await {
                                                Ok(_f) => {
                                                    {
                                                        let mut etalons_images_ = etalons_images.lock().await;
                                                        let ci = etalons_images_.entry(update.host.clone()).or_insert(
                                                             StreamRgb {
                                                                host: update.host.clone(),
                                                                id: update.id.clone(),
                                                                rgb: _f.to_vec(),
                                                                width: update.width.clone(),
                                                                height: update.height.clone()
                                                            }
                                                        );
                                                        *ci = StreamRgb {
                                                            host: update.host.clone(),
                                                            id: update.id.clone(),
                                                            rgb: _f.to_vec(),
                                                            width: update.width.clone(),
                                                            height: update.height.clone()
                                                        }
                                                    }
                                                },
                                                Err(_) => {}
                                            }
                                            match fs::read(etalon_file).await {
                                                Ok(_f) => {
                                                    let buf_prev_n = _f.to_vec();
                                                    //загружаем картинки(эталон и текущую) в тензоры для анамалий
                                                    let tensor1 = frame_to_tensor(&buf_curr_resized.clone(), 320, 240).unsqueeze(0);
                                                    let tensor2 = frame_to_tensor(&buf_prev_n.clone(), 320, 240).unsqueeze(0);
                                                    //отдаем их модели для анамалий
                                                    let t1 = model_anomaly.forward_ts(&[tensor1]).unwrap();
                                                    let t2 = model_anomaly.forward_ts(&[tensor2]).unwrap();
                                                    //cмотрим разницу
                                                    let dist = (&t1 - &t2).norm().double_value(&[]);
                                                    println!("Значение на выходе модели проверки анамалий: {}", dist);
                                                    //тоже самое для ракурса
                                                    let tensor1 = frame_to_tensor(&buf_curr_resized.clone(), 320, 240).unsqueeze(0);
                                                    let tensor2 = frame_to_tensor(&buf_prev_n.clone(), 320, 240).unsqueeze(0);
                                                    let t1 = model_rakurs.forward_ts(&[tensor1]).unwrap();
                                                    let t2 = model_rakurs.forward_ts(&[tensor2]).unwrap();
                                                    let t1 = normalize(&t1);
                                                    let t2 = normalize(&t2);
                                                    let dist_rakurs = 1.0-((&t1 * &t2).sum(Kind::Float).double_value(&[]));
                                                    println!("Значение на выходе модели проверки ракурса: {}", dist_rakurs);
                                                    {//записываем метрики
                                                        let mut m = metrics.lock().await;
                                                        let val_ = format!("{:.5}", dist.to_string());
                                                        let key = format!("anomaly{{host=\"{}\", Stream=\"{}\"}}", update.host.clone(), update.id);
                                                        let m_ = m.entry(key).or_insert(val_.clone());
                                                        *m_ = val_; 
                                                        let val_ = format!("{:.5}", dist_rakurs.to_string());
                                                        let key = format!("rakurs{{host=\"{}\", Stream=\"{}\"}}", update.host.clone(), update.id);
                                                        let m_ = m.entry(key).or_insert(val_.clone());
                                                        *m_ = val_;
                                                    }
                                                },
                                                Err(_) => {}
                                            }
                                        }
                                    }
                                }
                            });
                        }
                    }
                }
            });
        }
        {//async update metrics from ffmpeg - поток для обновления метрик битрейта
            tokio::spawn({
                let metrics = metrics.clone();
                async move {
                    while let Some(update) = rx_stats.recv().await {
                        if update.rate != 0.0 {
                            let mut m = metrics.lock().await; //блокируем метрики
                            let rate = format!("{:.3}", update.rate);
                            let stype = match update.media_type {
                                ffmpeg_next::media::Type::Video => "Video",
                                ffmpeg_next::media::Type::Audio => "Audio",
                                _ => ""
                            }.to_string(); //тип потока - видео или аудио
                            //ключ
                            let key = format!("bitrate{{host=\"{}\", MediaType=\"{}\", Stream=\"{}\"}}", update.host, stype, update.id);
                            let m_ = m.entry(key).or_insert(rate.clone()); //так же извлекаем метрику и если нету добавляем со значением rate
                            *m_ = rate; //обновляем если уже была
                        }
                    }
                }
            });
        }
        //тут запускается поток, который будет подключаться к rtsp потоку камеры, считать битрейт и отправлять метрики битрейта в другой поток и раз в 10 секунд снимать "скриншот" и отправлять в поток для аналитики через нейронки.
        { //rtsp(ffmpeg-next)
            tokio::spawn({
                let cam = cam.clone();
                async move {
                    //составляем url rtsp потока из параметров камеры
                    //получится например такая ссылка - rtsp://user2:E7L7M42Kdb@10.172.242.51:554/av0_0
                    let rtsp_url = format!("rtsp://{}:{}@{}:{}{}", cam.username.as_str(), cam.password.as_str(), cam.ip.as_str(), cam.rtsp_port, cam.rtsp_uri.as_str());
                    //запускаем функцию из этого модуля
                    rtsp::get_stream(cam.ip.clone(), tx_stats, tx_rgb, rtsp_url.clone(), cam.rtsp_transport.clone()).await;
                }
            });
        }
    }
    //в основном потоке запускается веб сервер для вывода метрик
    let app = Router::new().route(config.metrics.uri.as_str(), get(move |query| {
        let metrics = metrics.clone();
        let current_images = current_images.clone();
        let etalons_images = etalons_images.clone();
        http_handler(metrics, current_images, etalons_images, query)
	}));

    let addr: SocketAddr = config.metrics.bind.parse().expect("Unable to parse socket address");
	println!("HTTP metrics on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Deserialize, Debug)]
struct Params {
    host: Option<String>,
    host_etalon: Option<String>
}
async fn http_handler(metrics: ArcMetric, current_images: Arc<Mutex<HashMap<String, StreamRgb>>>, etalons_images: Arc<Mutex<HashMap<String, StreamRgb>>>, Query(params): Query<Params>) -> impl IntoResponse {
    match params.host_etalon {
        Some(host) => {
            //если указан хост - выводим картинку эталон
            let mut rgb_ = StreamRgb {
                host: "".to_string(),
                id: 0,
                rgb: Vec::new(),
                width: 0,
                height: 0
            };
            {
                let etalons_images_ = etalons_images.lock().await;
                if let Some(rgb) = etalons_images_.get(&host) {
                    rgb_ = rgb.clone();
                }
            }
            if rgb_.host != "" {
                let img_buffer: RgbImage = ImageBuffer::from_raw(rgb_.width as u32, rgb_.height as u32, rgb_.rgb.clone()).expect("Неверный размер буфера");
                let img = DynamicImage::ImageRgb8(img_buffer);
                let mut png_bytes = Vec::new();
                {
                    let mut cursor = std::io::Cursor::new(&mut png_bytes);
                    img.write_to(&mut cursor, image::ImageFormat::Png).expect("PNG encode failed");
                }
                let mut headers = HeaderMap::new();
                headers.insert("content-type", "image/png".parse().unwrap());
                (headers, Bytes::from(png_bytes))
            } else {
                let mut headers = HeaderMap::new();
                headers.insert("content-type", "text/plain; charset=utf-8".parse().unwrap());
                (headers, Bytes::from("".to_string().into_bytes()))
            }
        },
        None => {
            match params.host {
                Some(host) => {
                    //если указан хост - выводим картинку
                    let mut rgb_ = StreamRgb {
                        host: "".to_string(),
                        id: 0,
                        rgb: Vec::new(),
                        width: 0,
                        height: 0
                    };
                    {
                        let current_images_ = current_images.lock().await;
                        if let Some(rgb) = current_images_.get(&host) {
                            rgb_ = rgb.clone();
                        }
                    }
                    if rgb_.host != "" {
                        let img_buffer: RgbImage = ImageBuffer::from_raw(rgb_.width as u32, rgb_.height as u32, rgb_.rgb.clone()).expect("Неверный размер буфера");
                        let img = DynamicImage::ImageRgb8(img_buffer);
                        let mut png_bytes = Vec::new();
                        {
                            let mut cursor = std::io::Cursor::new(&mut png_bytes);
                            img.write_to(&mut cursor, image::ImageFormat::Png).expect("PNG encode failed");
                        }
                        let mut headers = HeaderMap::new();
                        headers.insert("content-type", "image/png".parse().unwrap());
                        (headers, Bytes::from(png_bytes))
                    } else {
                        let mut headers = HeaderMap::new();
                        headers.insert("content-type", "text/plain; charset=utf-8".parse().unwrap());
                        (headers, Bytes::from("".to_string().into_bytes()))
                    }
                }
                None => {
                    let m = metrics.lock().await;
                    let mut out = String::new();
                    //в цикле метрики выводятся
                    for (m_, v_) in m.iter() {
                        out.push_str(&format!("{} {}\n",  m_, v_));
                    }
                    let mut headers = HeaderMap::new();
                    headers.insert("content-type", "text/plain; charset=utf-8".parse().unwrap());
                    (headers, Bytes::from(out.into_bytes()))
                }
            }
        }
    }
}
