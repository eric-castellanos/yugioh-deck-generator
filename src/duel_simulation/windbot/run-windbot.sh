#!/bin/bash

# Resolve WIND_HOST to IP if it's not already an IP address
if [[ "$WIND_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  HOST="$WIND_HOST"
else
  HOST=$(getent hosts "${WIND_HOST:-multirole}" | awk '{ print $1 }')
fi

PORT=${WIND_PORT:-7922}
DECK_FILE=${DECK_FILE}
SERVER_MODE=${SERVER_MODE:-false}

echo "Connecting to host: $HOST, port: $PORT"
echo "Using Deck: $DECK_FILE"
echo "ServerMode: $SERVER_MODE"

exec mono /windbot/WindBot.exe \
  ServerMode=$SERVER_MODE \
  Host=$HOST \
  Port=$PORT \
  DeckFile=$DECK_FILE

