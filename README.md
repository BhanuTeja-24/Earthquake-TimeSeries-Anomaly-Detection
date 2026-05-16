# Earthquake anomaly detection

This is my MSc dissertation project for CMP-L016 Deep Learning Applications at the University of Roehampton. The brief was Project #25 — build a deep learning model to detect earthquakes in seismic time-series data, treat it as an anomaly detection problem, and deploy it as a working web app.

Milestone2:

Milestone3:


YT link: https://youtu.be/yZNs8eoHcI0?si=rd_NJb1zkcNANKI_

Bhanu Teja Kunchapu (A00088365)

## What it does

You give the model a 10-second three-component seismic recording. It tells you whether the recording contains an earthquake. That's it.

What's interesting is *how* it does that. The model has never seen an earthquake during training — only background seismic noise. At inference time, the model tries to reconstruct whatever recording you give it. If it can reconstruct it well, it's noise (the model has seen plenty of that). If it can't, it's an anomaly, and most likely an earthquake.

This is anomaly detection, not earthquake prediction. The model doesn't tell you when the next earthquake will happen — that problem is essentially unsolved in seismology. It tells you whether an earthquake has already occurred in a recording you give it.

## Results

On the synthetic test data that ships with this repo:

- AUROC: 1.000
- AUPRC: 1.000
- F1 at the operating threshold: 0.998
- Earthquake-to-noise score separation: about 14x
- Training time on a free Colab T4: roughly 2 minutes

The synthetic numbers are an upper bound. The synthetic data is cleaner than real-world recordings, so on the real STEAD dataset I'd expect AUROC closer to 0.85–0.92, which is what other unsupervised work in this space reports. The contribution of this project isn't the headline number on synthetic data — it's the methodology, especially the top-k anomaly score, which I explain below and ablate in the report.

## Quick start

Easiest path is Google Colab:

1. Open `notebooks/Milestone_2_A00088365.ipynb` in Colab.
2. Set runtime to T4 GPU (free tier works fine).
3. Run all cells. Takes about 2 minutes. Trains both models, saves checkpoints.
4. Open `notebooks/Milestone_3_A00088365.ipynb` in the same session.
5. Run all cells. Evaluates the models, prints the metrics, and launches a Gradio demo with a public share link valid for 72 hours.

If you want to run it locally:

```bash
git clone https://github.com/YOUR_USERNAME/earthquake-anomaly-detection.git
cd earthquake-anomaly-detection
pip install -r requirements.txt
jupyter notebook
```

You'll need Python 3.10 or newer.

## The architecture, briefly

A Conv1D denoising autoencoder. The encoder is three Conv1d layers with stride 2, taking 1024 samples down to a feature map of 128 timesteps with 128 channels. That gets flattened and projected to a 16-dimensional bottleneck. The decoder mirrors this with transposed convolutions, getting back to the original shape. BatchNorm and GELU activations throughout.

I also trained a Conv1D variational autoencoder as a comparison model, with the same encoder/decoder shape but a stochastic Gaussian bottleneck and ELBO training objective. This was to keep some connection to the OmniAnomaly paper that the project brief cited.

I tried an LSTM autoencoder first, because that's what OmniAnomaly uses. It failed — on 1024-step inputs, the LSTM couldn't compress enough information through the bottleneck and the decoder just learned to output the channel mean. Switching to convolutions fixed it. The full story is in section III of the report.

## The top-k score (this is the key bit)

Most anomaly detection papers using autoencoders use mean reconstruction error over the whole window as the anomaly score. That doesn't work well here.

Earthquakes are *localised* in time. In a 1024-sample window containing an earthquake, only about 200 samples — the P-wave and S-wave arrival regions — carry the diagnostic information. The other 824 samples are still just background noise, and the model reconstructs them fine. If you average reconstruction error over the whole window, the spike from the earthquake gets diluted by all the surrounding "normal" error.

The fix is to take the mean of just the *top-k* largest per-timestep errors, with k=10. This preserves the spike. On identical model weights, this single change took AUROC from 0.94 to 1.00. The ablation table in section X of the report shows it's the largest single contributor to performance — bigger than BatchNorm, GELU, the tight bottleneck, or the denoising criterion.

It's a small change. About one line of code. I think it's the most useful thing in this project to anyone working on similar transient-anomaly problems.

## What's in this repo

```
.
├── README.md                                                  this file
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── Milestone_2_A00088365.ipynb                            data prep + model + training
│   └── Milestone_3_A00088365.ipynb                            testing + Gradio app
│
├── artifacts/                                                  empty by design — generated by running the notebooks
│   └── .gitkeep
│
├── sample_data/
│   └── sample_quake_window.npy                                 small test file for the Gradio upload tab
│
└── docs/
    ├── ARCHITECTURE.md                                         extended design notes
    └── YOUTUBE_SCRIPT.md                                       script for the demo video
```

> **Note:** The IEEE-format report (PDF) is submitted separately and will be added to `docs/` after final revisions.

The `artifacts/` folder is empty because the model checkpoints and figures are regenerated every time you run the notebooks. Keeping them out of git means the repo stays small and you can verify everything reproduces from scratch.

## Using real STEAD data

The notebooks default to a synthetic seismogram generator so everything runs end-to-end without requiring a 4 GB download. If you want to swap in the real Stanford Earthquake Dataset:

1. Download chunks from https://github.com/smousavi05/STEAD. For a sanity check, chunk 1 plus the noise chunk is enough.
2. In Milestone 2, section 3.2, uncomment the `load_stead()` call and point it at your HDF5 and CSV files.
3. Re-run everything from the top.

The detection log table in section 5.5 of Milestone 3 will automatically pick up the STEAD magnitude column once real data is loaded.

## Things this project doesn't do

In the interest of being honest about scope:

- **It doesn't predict future earthquakes.** That's earthquake forecasting and it's an essentially unsolved problem in seismology. I'm not trying to solve that here.
- **It doesn't do real-time early warning.** Operational early warning systems work on streaming continuous data and need infrastructure beyond what this project includes.
- **It can't distinguish earthquakes from instrumental glitches.** Both score as anomalies. Adding a downstream classifier on the latent vector to separate the two would be a reasonable next step but I didn't have time.
- **It's single-station.** Real seismic networks have many stations and you can do much better by fusing them. That's a graph neural network problem and out of scope for this project.

I wrote a longer version of this in section X-D of the report.

## Demo video

The walkthrough video is linked in the assignment submission. The script I followed for recording it is in `docs/YOUTUBE_SCRIPT.md` if you want to see what each section is about without watching the whole thing.

## References

The full reference list is at the end of the report. The papers that mattered most for this project:

- Mousavi et al., "STanford EArthquake Dataset (STEAD)," IEEE Access, 2019 — the dataset
- Su et al., "Robust anomaly detection for multivariate time series through stochastic recurrent neural network" (OmniAnomaly), KDD 2019 — the framing the brief asked me to use
- Mousavi et al., "Earthquake transformer," Nature Communications, 2020 — the supervised state of the art on STEAD
- Vincent et al., "Extracting and composing robust features with denoising autoencoders," ICML 2008 — for the denoising criterion
- Loshchilov and Hutter, "Decoupled weight decay regularization" (AdamW), ICLR 2019 — for the optimiser

## License

MIT. See LICENSE. The STEAD dataset has its own license, see the link above.

## Citation

If for some reason you want to cite this:

```
Kunchapu, B. T. (2025). Earthquake Time-Series Anomaly Detection Using a
Conv1D Denoising Autoencoder. MSc Dissertation, University of Roehampton,
CMP-L016 Deep Learning Applications.
```
