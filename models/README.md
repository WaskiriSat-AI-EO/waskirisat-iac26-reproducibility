# models/

Binaries (`.tflite`, `.pt`) are not stored in git. `MANIFEST.json` is the source of truth:
id, sha256, size, owner, download URL. After publishing a binary (GitHub Release / Zenodo),
fill `url` and run `make verify-models`.

```
models/
├── MANIFEST.json
├── cloud/      # Leyla:   fp32 checkpoint, int8 .tflite, training + quantization reports
└── thermal/    # Osvaldo: frp20000/ (reports + sha256 already here)
```

Adding an artifact: copy reports here, add an entry to `MANIFEST.json`, open a PR.
