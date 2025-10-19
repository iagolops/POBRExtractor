# Photometric Object and Background Reduction Extractor (POBRExtractor)

![POBRExtractor Logo](logo/POBRExtractor-removebg.png)

**POBRExtractor** is a Python-based pipeline designed to perform source detection and photometric measurements on multi-band astronomical images.  
It takes as input several images of the same sky region across different filters and produces an object catalog containing astrometric and photometric information.

This project was developed as part of the course **“Tratamento de Dados Astronômicos”** at the *Observatório do Valongo – UFRJ*, under the supervision of **Bruno Morgado**.  
This code is designed as a simplified educational prototype rather than a full production pipeline. Please don't take it serious.

---

## 💻 Running POBRExtractor

```bash
python src/POBRExtractor.py data/input.yaml 
```
---

## 🧠 Workflow Overview

The diagram below summarizes the internal workflow of the pipeline:

![POBRExtractor Workflow](logo/POBRExtractor%20Workflow.png)

---

## 🌌 Example of source detection

HSC's coadd image (i-band)

![POBRExtractor Workflow](data/plots/residual_vs_raw.png)

## 📚 Citation / Acknowledgment

If you use this code (PLEASE DON'T), please cite:

> Lopes, Iago (2025). *Photometric Object and Background Reduction Extractor (POBRExtractor)* (It doensn't exists 🫠).  
> Course Project, Observatório do Valongo – UFRJ.  

---

## 🧩 Notes

> ⚠️ **Disclaimer:**  
> This project is an **educational prototype** created for learning and demonstration purposes.  
> It is **not intended for production-level data processing**.
---

