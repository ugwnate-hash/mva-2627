# TD1: Geometry Processing

Keep `TD1_Q.ipynb`, `requirements.txt`, the mesh files, `img/`, and `plot_utils/` together. The notebook bootstrap locates this folder from either the TD1 directory or the repository root.

## VS Code

Open this `TD1` folder in VS Code, install the Python and Jupyter extensions, and create an environment in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open `TD1_Q.ipynb`, select `.venv` as its kernel, and run from the first cell.

## Google Colab

GitHub's notebook preview is read-only; use Colab to execute the notebook. Run this setup in a Colab cell before the notebook's first setup cell:

```python
!git clone https://github.com/ugwnate-hash/mva-2627.git /content/mva-2627
%cd /content/mva-2627/MVA13679-Geometry-Processing-GDL/tp/TD1
%pip install -r requirements.txt
```

The repository must be accessible to the Colab runtime. Do not put access tokens in the notebook.

## Exercise Status

This is a practical exercise sheet and still contains unimplemented or unfinished questions. The setup makes its files and dependencies available; it does not imply every exercise cell runs top-to-bottom.