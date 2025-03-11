## 환경 설정 방법

본 강의는 가상환경을 생성하여 진행합니다. 다음 명령어를 차례대로 실행하여 환경을 설정하세요.

### 1. 가상환경 생성

다음 명령어를 통해 Python 3.10을 사용하는 `ComputerVision`이라는 이름의 가상환경을 생성합니다.

```bash
conda create -n ComputerVision python=3.10
```

### 2. 가상환경 활성화

생성한 환경을 활성화합니다.

```bash
conda activate ComputerVision
```

### 3. Jupyter Kernel 등록

Jupyter Notebook에서 가상환경을 사용할 수 있도록 커널(Kernel)을 등록합니다.

```bash
pip install ipykernel
python -m ipykernel install --user --name ComputerVision --display-name ComputerVision
```

### 4. 필요 패키지 설치

강의에서 사용하는 필수 패키지를 설치합니다. (`requirements.txt` 파일이 있는 폴더에서 실행하세요.)

```bash
pip install -r requirements.txt
```

설치가 완료되면 Jupyter Notebook을 실행하여 `ComputerVision` 커널을 선택해 사용하면 됩니다.
