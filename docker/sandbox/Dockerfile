FROM zakuroai/sakura:vanilla

COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl
RUN apt install redis -y

RUN mkdir -p /opt/zakuro/bin && ln -s /opt/miniconda/envs/zak38/bin/python /opt/zakuro/bin/sakura
COPY docker/sandbox/static/sakura /usr/local/bin/sakura

RUN pip install lightning
RUN pip install --force-reinstall charset-normalizer==3.1.0 redis
RUN echo "nohup redis-server >/dev/null 2>&1 &" >> /entrypoint
RUN echo "/usr/local/bin/sakura" >> /entrypoint
RUN chmod +x /entrypoint

RUN mkdir -p /opt/zakuro/logs/
CMD [ "/entrypoint" ]