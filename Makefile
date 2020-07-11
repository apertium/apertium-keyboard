
all: apk

apk:
	gradle assembleDebug
	cp ./app/build/outputs/apk/debug/app-debug.apk .
