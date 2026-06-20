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
