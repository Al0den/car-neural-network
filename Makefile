car:
	CCFLAGS="-Ofast" python3 -m nuitka src/main.py --follow-imports --disable-console --macos-create-app-bundle --static-libpython=no -j8
perft:
	(echo '10\n' | python3 -m cProfile -o ./data/profile.txt src/main.py) && echo -e '\n' && (echo 'sort cumtime\nreverse\nstats' | python3 -m pstats ./data/profile.txt) | grep -v '<' | grep 'src/'
compile_shader:
	cp ./src/shaders/shaders.metal ../metal-python/PyMetalBridge/Sources/PyMetalBridge/shaders.metal && cp ./src/shaders/PyMetalBridge.swift ../metal-python/PyMetalBridge/Sources/PyMetalBridge/PyMetalBridge.swift && cd ../metal-python/PyMetalBridge && rm -rf .build && ./build_metal.sh && swift build -c release && cp .build/release/libPyMetalBridge.dylib ../../Car/src/shaders/ && cd ../../Car && mv src/shaders/libPyMetalBridge.dylib src/shaders/compiled_shader.dylib
perft_results:
	echo 'sort cumtime\nreverse\nstats' | python3 -m pstats ./data/profile.txt
results:
	@read -p "Enter the track: " user_input && \
	echo "You entered: $$user_input" && \
	echo "7\n0\n$$user_input" | python src/main.py && \
	echo "\n$$user_input\n" | python utils/ff1.py

run:
	osascript -e 'tell application "Terminal" to do script "cd Desktop/projects/Car && echo \"2\n8\n\" | python src/main.py"'
graph:
	osascript -e 'tell application "Terminal" to do script "cd Desktop/projects/Car && echo \"g\" | python utils/graph.py"'
term:
	osascript -e 'tell application "Terminal" to do script "cd Desktop/projects/Car"'
