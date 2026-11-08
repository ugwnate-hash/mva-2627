# TD2: Geometry Processing

Keep this folder together as the runnable unit. The notebook expects the sibling `data/`, `mesh_utils/`, and `plot_utils/` directories, and writes generated meshes to `results/`.

```text
TD2/
  TD2.ipynb
  requirements.txt
  data/
  mesh_utils/
  plot_utils/
  results/
```

## VS Code

Open this `TD2` folder in VS Code, install the Python and Jupyter extensions, and create an environment from PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open `TD2.ipynb`, select the `.venv` Python interpreter as its kernel, and run from the first cell. Keeping the workspace rooted at this folder makes the notebook's asset paths predictable.

## Google Colab

GitHub's notebook preview is read-only; use Colab to execute the notebook. Open the notebook in Colab, then run this setup in a cell before the notebook's first setup cell:

```python
!git clone https://github.com/ugwnate-hash/mva-2627.git /content/mva-2627
%cd /content/mva-2627/MVA13679-Geometry-Processing-GDL/tp/TD2
%pip install -r requirements.txt
```

The repository must be accessible to the Colab runtime; private repositories require an authenticated checkout. Do not put access tokens in the notebook.

The first notebook cell detects the TD2 project root and enables Colab widgets when it is running in Colab. Keep `data/`, `mesh_utils/`, and `plot_utils/` in the same checked-out folder as the notebook.

## Important

The notebook is an exercise sheet and still contains unimplemented questions/TODOs. Environment setup makes its imports and files portable; it does not fill in those exercise solutions or guarantee that every cell runs top-to-bottom yet.