# FEM_I_Notebooks

Balken FEM for Binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/steffenbeese/FEM_I_Notebooks/main?urlpath=%2Fdoc%2Ftree%2FNotebook_BalkenFEM.md)


Stab FEM for Binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/steffenbeese/FEM_I_Notebooks/main?urlpath=%2Fdoc%2Ftree%2FNotebook_StabFEM.ipynb)


# Notes Praktikum 24.04.25

- Python Notebook zur Stab FEM
- Das Python Modul StabFEM.py liegt bei
  - Anforderung an externen Abhängigkeiten ist gering
    - Standardpakete: numpy, matplotlib und sympy (für Herleitung der FEM)
- Intro: Jupyter Notebooks:
  - Jupyter Notebooks sind interaktive Python Notebooks
  - Sie können in der Cloud ausgeführt werden (z.B. myBinder, Google Colab, Sage Math)
  - Sie können lokal ausgeführt werden (z.B. Anaconda, JupyterLab, VS Code)
  - links die Leiste:
    - Datei: Dateioperationen
    - Kernels: welche Kernel sind aktiv
    - Struktur: TOC
    - Extension Manager

  - Neues Notebook erstellen:
    - Variable a=3, b=5 erstellen
    - c=a+b
    - print(c)
    - liste erstellen: l1 = [1,2,3,4,5] und l2 = [6,7,8,9,10] und l3 = l1+l2
    - numpy array erstellen: a1 = np.array([1,2,3,4,5]) und a2 = np.array([6,7,8,9,10]) und a3 = a1+a2
    - Funktion plotten: sin(x); x = np.linspace(0,2*np.pi,100); y = np.sin(x); plt.plot(x,y)
    - Schleife programmieren:
       for i in range(len(x)):
          print(f'i= {i}, x= {x[i]}, y= {y[i]}')

- Das Notebook muss sukzessiv ausgeführt werden, da die Abhängigkeiten nacheinander eingeführt werden, da wo sie benötigt werden 
- Das Notebook ist zweigeteilt:
  - 1. Teil: Herleitung der FEM für den Dehnstab mit sympy
  - 2. Teil: Anwendung der StabFEM auf verschiedene Beispiele

