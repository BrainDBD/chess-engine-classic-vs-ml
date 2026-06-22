# Evaluarea pozițiilor finale în jocul de șah

Proiect de licență realizat de **Luparu Ioan-Teodor**, Facultatea de Matematică și Informatică, Universitatea din București.

Proiectul compară mai multe metode de evaluară a pozițiilor finale (endgame positions).

## Instalare

Clonați repository-ul:

```sh
git clone https://github.com/BrainDBD/chess-engine-classic-vs-ml.git
```

## Fișiere externe

Proiectul folosește următoarele resurse externe:

| Resursă | Utilizare | Stare | Sursă |
| --- | --- | --- | --- |
| Tabele Syzygy WDL/DTZ (3–6 piese) | etichete de referință pentru poziții | de descărcat separat | [tablebase.lichess.ovh](https://tablebase.lichess.ovh/tables/standard/) |
| Fathom | interogarea tabelelor Syzygy | integrată direct în cod | [jdart1/Fathom](https://github.com/jdart1/Fathom/tree/master) |
| FastChess `v1.8.0-alpha` | experimentele practice (meciuri între motoare) | program extern | [Disservin/fastchess](https://github.com/Disservin/fastchess/releases/tag/v1.8.0-alpha) |

## Structura proiectului

```text
chess-engine-classic-vs-ml/
├── src/                    # implementarea completă în C++
│   ├── core/               # motorul de bază și evaluarea clasică
│   ├── net/                # modelul MLP exportat ca header C++
│   └── syzygy/             # interogarea tabelelor Syzygy (clasă ce integrează librăria Fathom)
├── ml/                     # codul Python pentru antrenare și evaluare
│   ├── endgame_features.py # calcularea caracteristicilor dintr-o poziție dată
│   ├── extract_dataset.py  # extragerea setului de date dintr-un fișier PGN
│   ├── train_endgame.py    # antrenarea modelului (PyTorch)
│   ├── evaluate.py         # analiză și generarea figurilor
│   └── plots/              # scripturile care generează figurile
│       ├── plot_confusion_matrix_wdl.py
│       ├── plot_feature_ablation.py
│       └── create_heatmap.py
└── figures/                # PDF-urile generate folosite în document
```

## Comenzi folosite în realizarea figurilor experimentale

> Comenzile se rulează din folderul `ml/`.

**Generarea setului de date**
```sh
python extract_dataset.py dataset.pgn --out endgame.jsonl \
    --max-pieces 6 --quiet-only --opening-skip 8 \
    --syzygy "tablebases/3-4-5-wdl,tablebases/6-wdl" --syzygy-only
```
Comenziile de mai jos folosesc două seturi de date: `endgame.jsonl` a fost creat din fișierul PGN luat din luna Aprilie 2026, iar `may_static-test.jsonl` este setul de date obținut din fișierul lunii Mai, filtrat cu ajutorul script-ului `ml/sample_static_test.py`.

**Antrenarea modelului**
```sh
python train_endgame.py endgame.jsonl --max-samples 1000000 --epochs 20
```

**Căutarea pragurilor de decizie**
```sh
python ordinal_threshold_sweep.py endgame.jsonl net_ordinal.pt
```

**Matricile de confuzie pe setul de testare** (`cm_may_test.pdf` + `cm_may_test_default.pdf`)
```sh
python evaluate.py may_static-test.jsonl --mode balanced --load net_ordinal.pt \
    --val-frac 1.0 --threshold 999 --t-draw 0.35 --t-win 0.65 \
    --plot-out figures/cm_may_test.pdf --plot-normalize rows
```

**Graficul de ablație** (`ablation_full.pdf`)
```sh
python evaluate.py endgame.jsonl --mode ablation --load net_ordinal.pt > ablation_full.txt
python plots/plot_feature_ablation.py ablation_full.txt --out figures/ablation_full.pdf
```

**Verificarea coliziunilor între caracteristici**
```sh
python feature_collision.py endgame.jsonl --val-frac 0.1 --seeds 42 --load net_ordinal.pt
```

**Tabelele PST ale regelui** (`king_pst_heatmaps.pdf`)
```sh
python plots/plot_heatmap.py
```
