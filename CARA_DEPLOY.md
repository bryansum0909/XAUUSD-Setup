# Cara Deploy Bot ke Cloud (GitHub Actions) — tanpa laptop

Bot akan jalan di server GitHub **tiap jam otomatis**. Laptop boleh mati.
Token kamu disimpan terenkripsi di GitHub (Secrets) — tidak pernah ada di kode.

## LANGKAH 1 — Buat repository di GitHub (web, 2 menit)
1. Buka https://github.com → login → klik tombol **+** (kanan atas) → **New repository**
2. Repository name: `xauusd-signal-bot` (bebas)
3. Pilih **Private** (disarankan) → **Create repository**
4. GitHub tampilkan halaman "…or push an existing repository". **Biarkan terbuka**, salin URL-nya
   (contoh: `https://github.com/NAMAKAMU/xauusd-signal-bot.git`)

## LANGKAH 2 — Upload folder cloud_deploy ke repo (pilih SATU cara)

**Cara A — Git command line** (jalankan di folder `cloud_deploy`):
```
git init
git add .
git commit -m "deploy xauusd signal bot"
git branch -M main
git remote add origin https://github.com/NAMAKAMU/xauusd-signal-bot.git
git push -u origin main
```
(Saat diminta login, pakai username GitHub + Personal Access Token sebagai password —
buat token di github.com → Settings → Developer settings → Personal access tokens.)

**Cara B — GitHub Desktop** (lebih mudah, kalau terpasang):
File → Add local repository → pilih folder `cloud_deploy` → Publish repository.

**Cara C — Web upload:** di repo baru → "uploading an existing file" → drag folder
`cloud_deploy` isinya (termasuk folder `.github` dan `src`) → Commit.

## LANGKAH 3 — Tambah 2 Secret (token & chat id)
Di repo GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
1. Name: `TG_BOT_TOKEN`  · Secret: (tempel token BotFather-mu) → Add secret
2. Name: `TG_CHAT_ID`   · Secret: (tempel chat id-mu) → Add secret
(Opsional: `TG_BALANCE` = 10000, `TG_RISK` = 0.5)

## LANGKAH 4 — Aktifkan & tes
1. Di repo → tab **Actions** → kalau ada tombol "I understand… enable", klik.
2. Pilih workflow **"XAUUSD H1 Signal Bot"** → **Run workflow** (tombol kanan) → jalankan manual sekali.
3. Tunggu ~1-2 menit. Kalau ada breakout → pesan masuk Telegram. Kalau tidak → normal
   (log Actions menunjukkan "no new signal").

Setelah itu bot **jalan sendiri tiap jam**, laptop boleh mati. ✅

## Catatan
- Private repo gratis ~2000 menit/bulan (cukup untuk per-jam). Kalau kena limit, ubah repo jadi Public (menit tak terbatas; token tetap rahasia).
- Data live = Yahoo GC=F 1h + DXY. Verifikasi level di chart broker sebelum entry.
- Demo & manual saja. Bukan saran keuangan.
