GCPUSAGE='Runs tasks at Google Cloud Platform.
Provide task after gcp like this: run.sh gcp my-task [parameter ...].
If no tasks provided (run.sh gcp) it syncs remote and local files (gets results).'


if [[ -z ${AISCIBB_GCP_FLAG-} ]] ; then
	echo $GCPUSAGE
	# In local environment.
	[[ ! -z ${AISCIBB_GIT_TOKEN-} ]]  || ( echo "Please set AISCIBB_GIT_TOKEN environment variable (see https://github.com/settings/tokens)." ; exit 1 )
	[[ ! -z ${AISCIBB_GCP_SSH_USERHOST-} ]]  || ( echo "Please set AISCIBB_GCP_SSH_USERHOST environment variable (example: user@123.123.123.123)." ; exit 1 )

	echo Uploading files required to run commands...
	scp -q ~/.gitconfig scp://$AISCIBB_GCP_SSH_USERHOST/.gitconfig
	scp -q ./run.sh scp://$AISCIBB_GCP_SSH_USERHOST/run.sh
	if [[ $# -gt 0 ]]; then
		echo Calling GCP to run a task...
		ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp $( git rev-parse --abbrev-ref HEAD ) "$AISCIBB_GIT_TOKEN" "$@"
	fi

	echo Getting results
	ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp

	echo Syncing artifacts from GCP...
	# TODO: configure artifacts dir at remote.
	rsync --info=progress2 -r $AISCIBB_GCP_SSH_USERHOST:~/artifacts/\* $AISCBB_ARTIFACTS_DIR
	echo Done. Artifacts at $AISCBB_ARTIFACTS_DIR
else
	# In remote environment.
	# If all prerequisits met:
	mkdir -vp $AISCBB_ARTIFACTS_DIR
	mkdir -vp $AISCBB_DATA_DIR

	C_ID_FILE=./aiscbb_container_id.cid
	if [[ $# -eq 0 ]]; then
		echo "Containers (tasks):"
		sudo docker ps
		if [[ -e $C_ID_FILE ]]; then
			CID=$( cat $C_ID_FILE ) 
			if sudo docker ps | grep $CID ; then 
				C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
				sudo docker logs -f $CID |& tee $C_LOG_FILE
			fi
		fi
	else
		echo [GCP] Run a task...
		echo [GCP] Stoping current task if any.
		if [[ -e $C_ID_FILE ]] ; then 
			CID=$( cat $C_ID_FILE ) 
			if sudo docker ps | grep $CID ; then 
				echo There is $CID container running. Stoppping...
				C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
				sudo docker logs -f $CID &> $C_LOG_FILE
				sudo docker rm $CID
			fi
			rm $C_ID_FILE
		fi
		
		echo Clonning repository...
		GITBRANCH=${1-$GIT_MAIN_BRANCH_NAME}
		AISCIBB_GIT_TOKEN=$2
		TARGETDIR=~/aiscbbproj
		if [[ -e $TARGETDIR ]] ; then 
			echo Removing old repo directory
			rm -rf $TARGETDIR
		fi
		git clone -b $GITBRANCH "https://$AISCIBB_GIT_TOKEN@$GIT_REMOTE" $TARGETDIR 

		echo Building docker image...
		( cd $TARGETDIR ; sudo docker buildx build -t aiscbb . )

		echo Running container...
		shift 2  # Remove first two params for gcp.

		sudo docker container run \
			-e AISCBB_ARTIFACTS_DIR=/aiscbb_artifacts \
			-e AISCBB_DATA_DIR=/asicbb_data \
			-v $AISCBB_ARTIFACTS_DIR:/aiscbb_artifacts \
			-v $AISCBB_DATA_DIR:/asicbb_data \
			-v ~/.gitconfig:/etc/gitconfig \
			--cidfile="$C_ID_FILE" \
			--detach \
			aiscbb bash run.sh "$@"
		CID=$( cat $C_ID_FILE )
		C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
		sudo docker logs -f $CID |& tee $C_LOG_FILE

		[[ ! -e %TARGETDIR ]] || rm -dr $TARGETDIR
		echo At GCP. Finished.
	fi
fi