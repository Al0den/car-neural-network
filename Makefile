car:
	CCFLAGS="-Ofast" python3 -m nuitka src/main.py --follow-imports --disable-console --macos-create-app-bundle --static-libpython=no -j8
perft:
	(echo '10\n' | python3 -m cProfile -o ./data/profile.txt src/main.py) && echo -e '\n' && (echo 'sort cumtime\nreverse\nstats' | python3 -m pstats ./data/profile.txt) | grep -v '<' | grep 'src/'
