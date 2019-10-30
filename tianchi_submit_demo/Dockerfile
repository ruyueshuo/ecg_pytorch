FROM  registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
#FROM  registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-py3
ADD . /competition
WORKDIR /competition
RUN pip --no-cache-dir install -r requirements.txt
CMD ["sh", "run.sh"]

