# Aproximación de filtros digitales IIR en punto fijo

## Requisitos

Se requiere Python 3.6 o superior. Las instrucciones para su instalación
varían según el sistema operativo:

- [Linux](https://docs.python-guide.org/starting/install3/linux/)
- [MacOS](https://docs.python-guide.org/starting/install3/osx/)
- [Windows](https://docs.python-guide.org/starting/install3/win/)

## Entorno

1. Crear un entorno virtual:

**Linux/MacOS**
```
python3 -m venv venv
```

**Windows**
```
python -m venv venv
```

2. Activar el entorno virtual:


**Linux/MacOS**
```
. venv/bin/activate
```

**Windows**
```
venv\Scripts\activate
```

3. Instalar dependencias dentro del entorno:

```
pip install --upgrade pip
pip install -r requirements.txt
```

El entorno virtual puede ser desactivado mediante `deactivate`.

## Uso

Estando dentro del entorno virtual, ejecutar:

```
jupyter notebook
```

Luego, ubicar y abrir `fwliir.ipynb` en la interfaz web. Si bien no es estrictamente necesario para su visualización, 
se recomienda [instalar la extensión](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) 
`Hide Input` para reducir la verbosidad del documento.
