use pnet::packet::icmp::{echo_request, IcmpPacket, IcmpTypes};
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::Packet;
use pnet::transport::{icmp_packet_iter, transport_channel, TransportChannelType, TransportProtocol};
use std::net::IpAddr;
use std::time::{Duration, Instant};
use pnet::packet::icmp::echo_reply::EchoReplyPacket;
use crate::ArcMetric;

pub async fn ping(host_: &str, metrics: ArcMetric) {
    let host: IpAddr = host_.parse().expect("Некорректный IP-адрес");
    let (mut tx, mut rx) = transport_channel(
        1024,
        TransportChannelType::Layer4(TransportProtocol::Ipv4(IpNextHeaderProtocols::Icmp)),
    ).expect("Не удалось открыть raw-сокет (нужен root)");

    let mut seq: u16 = 0;
    let identifier: u16 = (std::hash::Hasher::finish(&mut std::collections::hash_map::DefaultHasher::new()) % u16::MAX as u64) as u16;

    let mut sent: u64 = 0;
    let mut received: u64 = 0;
    let mut total_rtt: f64 = 0.0;
    let mut last_rtt_val: Option<f64> = None;
    let mut jitter_sum: f64 = 0.0;

    let mut iter = icmp_packet_iter(&mut rx);
    /* бесконечный цикл пинга */
    loop {
        /* всякие параметры пакета для запроса пинга */
        let mut buffer = [0u8; 64];
        let mut echo = echo_request::MutableEchoRequestPacket::new(&mut buffer).unwrap();
        echo.set_icmp_type(IcmpTypes::EchoRequest);
        echo.set_sequence_number(seq);
        echo.set_identifier(identifier);
        let checksum = pnet::packet::icmp::checksum(&IcmpPacket::new(echo.packet()).unwrap());
        echo.set_checksum(checksum);

        let send_time = Instant::now();
        /* отправка его */
        match tx.send_to(echo, host) {
            Ok(_) => {}
            Err(_) => {}
        }
        /* добавление в переменную сколько отправлено пакетов */
        sent += 1;

        let start = Instant::now();
        /* ожидание ответа */
        while Instant::now().duration_since(start) < Duration::from_secs(1) {
            if let Ok(Some((packet, host_reply))) = iter.next_with_timeout(Duration::from_millis(100)) {
                if host_reply == host {
                    if let Some(reply) = EchoReplyPacket::new(packet.packet()) {
                        if reply.get_identifier() == identifier && reply.get_sequence_number() == seq {
                            /* сохранение сколько прошло времени */
                            let rtt = Instant::now().duration_since(send_time).as_secs_f64() * 1.0; //1000.0;
                            /* сколько пакетов принято */
                            received += 1;
                            total_rtt += rtt;
                            if let Some(prev) = last_rtt_val {
                                jitter_sum += (rtt - prev).abs();
                            }
                            last_rtt_val = Some(rtt);
                            break;
                        }
                    }
                }
            }
        }
        /* посчет потерь */
        let loss = if sent > 0 {
            100.0 * (sent - received) as f64 / sent as f64
        } else {
            0.0
        };
        /* средний ртт - общеее время на поход пакета туда и обратно */
        let avg_rtt = if received > 0 {
            total_rtt / received as f64
        } else {
            0.0
        };
        /* посчет джиттера - среднее время между максимальным и минимальным таймингом */
        let jitter = if received > 1 {
            jitter_sum / (received - 1) as f64
        } else {
            0.0
        };

        /* после 10 пакетов отчищаем статистику */
        if sent > 10 {
            sent = 0;
            received = 0;
            jitter_sum = 0.0;
            last_rtt_val = None;
        }
        
        { /* заполняем метрики */
            let mut m = metrics.lock().await; /* блокируем массив метрик для других потоков */
        
            let loss_ = format!("{:.1}", loss); //значение метрики
            let key = format!("icmplosts{{host=\"{}\"}}", host_); //ключ метрики
            let m_ = m.entry(key).or_insert(loss_.clone()); //получение его из мессива и если ранее не было то присвоение значения
            *m_ = loss_; //обновление, если уже было

            let rtt_ = format!("{:.3}", avg_rtt); //тут так же для rtt
            let key = format!("icmprtt{{host=\"{}\"}}", host_);
            let m_ = m.entry(key).or_insert(rtt_.clone());
            *m_ = rtt_;
            
            let jitter_ = format!("{:.3}", jitter); //и для джиттера
            let key = format!("icmpjitter{{host=\"{}\"}}", host_);
            let m_ = m.entry(key).or_insert(jitter_.clone());
            *m_ = jitter_;
        } //тут он разблокируется, блок в скобках для того чтобы безопасно заблокировать/разблокировать, иначе может зависнуть
        seq = seq.wrapping_add(1); // увеличение id пакета
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}
