FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv

USER 1000
ENV HOME=/tmp

COPY /requirements.txt .

RUN pip install -r requirements.txt

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/openai:1.0.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/openai:1.0.0
