FROM lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860

WORKDIR /opt/experiment
COPY apply.py download.py versions.json LICENSE NOTICE ./
COPY patches/ patches/
ARG PROFILE=cached
LABEL deepseek-v41.profile=$PROFILE
RUN python3 apply.py --profile "$PROFILE"
COPY tests/ tests/
