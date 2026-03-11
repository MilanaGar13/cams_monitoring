use ffmpeg_next as ffmpeg;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::mpsc::Sender;
use crate::StreamStats;
use ffmpeg_next::{util::frame, software::scaling};
use crate::Mutex;
use crate::Arc;
use crate::StreamRgb;

pub async fn get_stream(host_: String, tx_stats: Sender<StreamStats>, tx_rgb: Sender<StreamRgb>, url: String, rtsp_transport: String) {
    ffmpeg::init().unwrap();
    let report_interval = Duration::from_secs(5);
    //бесконечный цикл для перезапуска в случае если завис поток
    loop {
        //переменная для проверки на зависший поток
        let last_packet: Arc<Mutex<Instant>> = Arc::new(Mutex::new(Instant::now()));
        let handle = tokio::task::spawn_blocking({
            let host_ = host_.clone();
            let url = url.clone();
            let last_packet = last_packet.clone();
            let tx_stats = tx_stats.clone();
            let tx_rgb = tx_rgb.clone();
            let rtsp_transport = rtsp_transport.clone();
            move || {
                let mut opts = ffmpeg::Dictionary::new();
                //параметры для входного потока ffmpeg
                opts.set("rtsp_transport", rtsp_transport.as_str());
                opts.set("stimeout", "5000000");
                opts.set("timeout", "5000000");
                opts.set("rw_timeout", "5000000");
                opts.set("reconnect", "1");
                let mut _st = Instant::now();
                let mut _startsendrgb = false;
                match ffmpeg::format::input_with_dictionary(&url, opts.clone()) {
                    Ok(mut ictx) => {
                        loop {
                            let mut stats: HashMap<usize, StreamStats> = HashMap::new();
                            //тут задаются параметры потоков(аудио/видео)
                            for (i, stream) in ictx.streams().enumerate() {
                                let params = stream.parameters();
                                let media = params.medium();
                                stats.insert(i, StreamStats {
                                    host: host_.to_string(),
                                    params: params.clone(),
                                    bytes: 0,
                                    id: i,
                                    last_report: Instant::now(),
                                    media_type: media,
                                    rate: 0.0
                                });
                            }

                            for (stream, packet) in ictx.packets() { //цикл получения собственно данных видео/аудио
                                { //обновление переменной, по сути оповещение что ничего не зависло
                                    let mut lp = last_packet.blocking_lock();
                                    *lp = Instant::now();
                                }
                                let idx = stream.index();
                                if let Some(stat) = stats.get_mut(&idx) {
                                    stat.bytes += packet.size() as u64; //подсчет битрейта
                                    if packet.is_key() && stat.media_type == ffmpeg_next::media::Type::Video { //если поток видео и это ключевой кадр - декодируем его в RGB и отправляем для аналитики
                                        if _st.elapsed() > Duration::from_secs(10) && !_startsendrgb { //это проверка чтобы раз в 10 секунд было, а не каждый кадр
                                            _startsendrgb = true;
                                            match ffmpeg::codec::Context::from_parameters(stat.params.clone()) {
                                                Ok(decoder) => {
                                                    match decoder.decoder().video() {
                                                        Ok(mut d) => {
                                                            let _ = d.send_packet(&packet);
                                                            let mut frame = ffmpeg::util::frame::Video::empty();
                                                            while d.receive_frame(&mut frame).is_ok() {
                                                                let mut rgb_frame = frame::Video::empty();
                                                                match scaling::Context::get(
                                                                    frame.format(),
                                                                    frame.width(),
                                                                    frame.height(),
                                                                    ffmpeg_next::format::Pixel::RGB24,
                                                                    frame.width(),
                                                                    frame.height(),
                                                                    scaling::Flags::BILINEAR)
                                                                {
                                                                    Ok(mut scaler) => {
                                                                        scaler.run(&frame, &mut rgb_frame).unwrap();
                                                                        let rgb_bytes = rgb_frame.data(0).to_vec();
                                                                        let streamrgb = StreamRgb {
                                                                            host: host_.to_string(),
                                                                            id: idx,
                                                                            rgb: rgb_bytes.clone(),
                                                                            width: frame.width(),
                                                                            height: frame.height()
                                                                        };
                                                                        tx_rgb.blocking_send(streamrgb).unwrap(); //отправка в поток с аналитикой
                                                                        _st = Instant::now();
                                                                        _startsendrgb = false;
                                                                    },
                                                                    Err(_) => {
                                                                        d.flush();
                                                                        _st = Instant::now();
                                                                        _startsendrgb = false;
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        Err(_) => {
                                                            _st = Instant::now();
                                                            _startsendrgb = false;
                                                        }
                                                    }
                                                }
                                                Err(_) => {
                                                    _st = Instant::now();
                                                    _startsendrgb = false;
                                                }
                                            }
                                            _st = Instant::now();
                                            _startsendrgb = false;
                                        }
                                    }
                                    let elapsed = stat.last_report.elapsed();
                                    if elapsed >= report_interval {
                                        let bits = (stat.bytes * 8) as f64;
                                        let rate = bits / elapsed.as_secs_f64(); //подсчет битрейта
                                        stat.rate = rate;
                                        stat.last_report = Instant::now();
                                        tx_stats.blocking_send(stat.clone()).unwrap(); //отправка битрейта в поток для задания метрик(прям тут нельзя задать метрики как в пинге, потому что работает синхронно и сам блокирует поток, не получится заблокировать метрики, поэтому через дополнительный поток)
                                        stat.bytes = 0;
                                    }
                                }
                            }
                            eprintln!("Reconnecting in 3 seconds...");
                            std::thread::sleep(std::time::Duration::from_secs(3));
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to open stream: {:?}", e);
                    }
                }
            }
        });
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let elapsed = {
                let lp = last_packet.lock().await;
                lp.elapsed()
            };
            if elapsed > Duration::from_secs(10) {
                eprintln!("No packets for 10s, restarting stream...");
                break;
            }
            if handle.is_finished() {
                eprintln!("FFmpeg task finished, reconnecting...");
                break;
            }
        }
        if !handle.is_finished() {
            handle.abort();
        }
        eprintln!("Reconnecting in 3 seconds...");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

