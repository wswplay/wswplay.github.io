FROM node:16.4.2-slim as base

LABEL "com.github.actions.name"="Biancheng Blog"
LABEL "com.github.actions.description"="Biancheng deploy"
LABEL "com.github.actions.icon"="upload-cloud"
LABEL "com.github.actions.color"="gray-dark"

LABEL "repository"="https://github.com/wswplay/wswplay.github.io"

RUN apt-get update && apt-get install -y git jq

COPY deploy.sh /deploy.sh
ENTRYPOINT ["/deploy.sh"]