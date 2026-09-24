# Deploy an toàn (máy chủ Docker) — không mất dữ liệu

Dữ liệu đang chạy (DB, số dư, vị thế OPEN, lịch sử) nằm trong Docker named
volume `btcusdt_data` (`/app/data` trong container). Rebuild giữ nguyên
volume nên **không mất dữ liệu** — miễn là làm đúng thứ tự dưới đây.

> ⛔ CẤM KỴ: không bao giờ chạy `docker compose down -v` hoặc
> `docker volume rm ...` — 2 lệnh này XÓA volume = mất DB.

Làm từng bước, bước nào OK mới sang bước tiếp theo.

---

## Bước 1. Kiểm tra tên volume

```bash
docker volume ls
```

Kết quả đúng: thấy 1 volume kết thúc bằng `btcusdt_data`
(ví dụ `tradingagents_btcusdt_data` — prefix là tên thư mục repo).
Ghi nhớ tên đầy đủ, dùng ở bước 2.

## Bước 2. Backup DB đang chạy

```bash
docker run --rm -v <tên-volume-đầy-đủ-ở-bước-1>:/data -v /tmp:/backup alpine cp /data/btcusdt_signals.db /backup/btcusdt_signals.db.bak
ls -la /tmp/btcusdt_signals.db.bak
```

Kết quả đúng: file `.bak` tồn tại, dung lượng > 0.
Bước này chỉ mất 1 phút nhưng cứu được toàn bộ dữ liệu nếu gõ nhầm lệnh.

## Bước 3. Pull code mới

```bash
git status --porcelain
```

Kết quả đúng: **không in ra dòng nào** (working tree sạch).
Nếu có dòng lạ hiện ra → DỪNG LẠI, chưa pull vội.

```bash
git pull
```

Ghi chú:

- `.env` không bị track nên pull không đụng API key.
- `data/btcusdt_signals.db` đang bị track trong git — pull có thể ghi đè
  file này **trên host**, nhưng container đọc DB từ volume nên
  **không ảnh hưởng dữ liệu đang chạy**.

## Bước 4. Build + chạy lại (giữ volume)

```bash
docker compose build
docker compose up -d
```

Kết quả đúng: container recreate nhưng volume giữ nguyên.
Daemon khởi động sẽ tự resume giám sát TP/SL cho vị thế đang OPEN.

## Bước 5. Kiểm tra sau deploy

```bash
curl localhost:8000/api/health
docker compose ps
```

Kết quả đúng: health trả `ok`, `ps` hiện container `healthy`.

Mở dashboard kiểm tra tay:

- [ ] Vị thế OPEN (nếu có) còn nguyên.
- [ ] Số dư / Equity đúng như trước deploy.
- [ ] Daemon Log chạy tiếp (không reset từ 0).

## Bước 6. Rollback nếu bản mới lỗi

```bash
git checkout <commit-cũ-đang-chạy-ổn>
docker compose up -d --build
```

Trường hợp xấu nhất (DB lỗi): copy file backup ở bước 2 trở lại volume
rồi restart container:

```bash
docker run --rm -v <tên-volume-đầy-đủ>:/data -v /tmp:/backup alpine cp /backup/btcusdt_signals.db.bak /data/btcusdt_signals.db
docker compose restart
```
