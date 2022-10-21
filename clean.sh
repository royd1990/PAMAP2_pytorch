docker stop $(docker ps -a -q  --filter ancestor=pamap2har:train)
docker rm $(docker ps -a -q  --filter ancestor=pamap2har:train)
docker rmi $(docker images --filter=reference=pamap2har:train)