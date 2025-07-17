#!/bin/bash

PORT=${WIND_PORT:-7911}
DECK_FILE=${DECK_FILE}
SERVER_MODE=${SERVER_MODE:-false}
NAME=${BOT_NAME:-WindBot}
EXPORT_PATH=${EXPORT_PATH:-/GameData}
HAND=${HAND:-2}
IS_TRAINING=${IS_TRAINING:-true}
SHOULD_UPDATE=${SHOULD_UPDATE:-true}
SHOULD_RECORD=${SHOULD_RECORD:-false}
IS_MCTS=${IS_MCTS:-false}

if [ "$SERVER_MODE" = "true" ]; then
  echo "🟢 WindBot running in SERVER MODE on port $PORT"
  exec mono /windbot/WindBot.exe \
    ServerMode=true \
    Port=$PORT \
    DeckFile=$DECK_FILE \
    ExportPath=$EXPORT_PATH \
    Name=$NAME \
    Hand=$HAND \
    IsTraining=$IS_TRAINING \
    ShouldUpdate=$SHOULD_UPDATE \
    ShouldRecord=$SHOULD_RECORD \
    IsMCTS=$IS_MCTS
else
  echo "🔵 WindBot running in CLIENT MODE, connecting to $WIND_HOST:$PORT"
  exec mono /windbot/WindBot.exe \
    Host=$WIND_HOST \
    Port=$PORT \
    DeckFile=$DECK_FILE \
    ExportPath=$EXPORT_PATH \
    Name=$NAME \
    Hand=$HAND \
    IsTraining=$IS_TRAINING \
    ShouldUpdate=$SHOULD_UPDATE \
    ShouldRecord=$SHOULD_RECORD \
    IsMCTS=$IS_MCTS
fi

