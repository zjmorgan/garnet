#!/bin/sh

DEBIAN=$(lsb_release -c -s)

cat << EOF > /etc/apt/sources.list.d/materiapps-${DEBIAN}.list
# for MateriApps LIVE!
deb http://exa.phys.s.u-tokyo.ac.jp/archive/MateriApps/apt/${DEBIAN} ${DEBIAN} main non-free contrib
deb-src http://exa.phys.s.u-tokyo.ac.jp/archive/MateriApps/apt/${DEBIAN} ${DEBIAN} main non-free contrib
EOF

apt-get -o Acquire::AllowInsecureRepositories=true update
apt-get -y --allow-unauthenticated install materiapps-keyring
apt-get update
