FROM registry.cn-hangzhou.aliyuncs.com/trt2022/dev

RUN mkdir /test_space

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
