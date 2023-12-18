car:
	CCFLAGS="-Ofast" python3 -m nuitka src/main.py --follow-imports --disable-console --macos-create-app-bundle --static-libpython=no -j8
