FROM zakuroai/sakura:vanilla

COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl && \
    rm -rf /tmp/*.whl

WORKDIR /workspace

USER foo

CMD ["sakura"]