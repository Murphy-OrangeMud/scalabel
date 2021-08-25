docker build . -t murphyorangemud/scalabel:try-communication

docker run -it -v "`pwd`/local-data:/opt/scalabel/local-data" --gpus all -p \
    8686:8686 -p 6379:6379 murphyorangemud/scalabel:try-communication



node app/dist/main.js --config ./local-data/scalabel/config.yml \
    --max-old-space-size=8192

torch-model-archiver -f --model-name test2_000030 --version 1.0 --export-path model_store --handler /opt/scalabel/scalabel/bot/handlers.py

torchserve --start --ncs --model-store=model_store --models=test2_000030.mar