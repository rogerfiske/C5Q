# 4. Runtime Environments

### Local Workstation (Detailed Specifications)

* **Operating System:** Windows 11 Home 64‑bit
* **CPU:** AMD Ryzen 9 6900HX (Rembrandt, 6nm technology) @ 56°C
* **RAM:** 64.0GB @ 2393MHz (40-39-39-77 timings)
* **Motherboard:** Shenzhen Meigao Electronic Equipment Co.,Ltd F7BAA (FP7) @ 20°C
* **Graphics:**
  - Primary: AMD Radeon RX 6600M (8176MB VRAM)
  - Integrated: AMD Radeon Graphics (512MB VRAM)
  - CrossFire: Disabled
* **Displays:**
  - VP2780 SERIES (2560x1440@59Hz)
  - ASUS PB278 (2560x1440@60Hz)
* **Storage:**
  - Samsung SSD 990 PRO 1TB (NVMe, Unknown)
  - KINGSTON OM8PGP41024Q-A0 1TB (SATA-2 SSD)
* **Execution Mode:** CPU‑only (EDA, prototyping, light training < 1 hour)
* **Thermal Management:** Adequate cooling with 56°C CPU temps under load
* **Memory Bandwidth:** Sufficient for 128-context window processing

### RunPod GPU

* **GPU:** NVIDIA H200
* **Mode:** CUDA training for NPL and Subset Diffusion.
* **Usage:** Offload heavy jobs via Docker; artifacts synced back.

---
