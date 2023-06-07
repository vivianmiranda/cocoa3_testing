if [ -z "${ROOTDIR}" ]; then
    echo 'ERROR ROOTDIR not defined'
    exit 1
fi

git add --all
sh $ROOTDIR/projects/.git_add.sh