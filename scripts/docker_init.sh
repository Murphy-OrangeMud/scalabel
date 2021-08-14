docker build . -t murphyorangemud/scalabel:try-communication

docker run -it -v "`pwd`/local-data:/opt/scalabel/local-data" -p \
    8686:8686 -p 6379:6379 murphyorangemud/scalabel:try-communication



node app/dist/main.js --config ./local-data/scalabel/config.yml \
    --max-old-space-size=8192