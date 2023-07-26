FROM zakuroai/compute:vanilla

COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl