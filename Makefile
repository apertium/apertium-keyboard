
all: apk

apk:
	gradle assembleDebug
	cp ./app/build/outputs/apk/debug/app-debug.apk apertium-keyboard.apk

clean:
	gradle clean
	rm -rf app/build
	rm -rf .gradle
