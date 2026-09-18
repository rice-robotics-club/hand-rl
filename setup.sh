### Python Environment Setup
echo "======================="
echo "CLAYS AWESOME SETUP SCRIPT"
echo "======================="
sleep 0.5
echo "Hi! welcome to the awesome setup script."
## uv setup
# check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "hello :) looks like uv isn't on your machine yet, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed. nice job!"
fi
sleep 0.5

# set up uv environment
echo "setting up our python environment... :)"
uv sync
sleep 0.5

uv run cowsay -c trex -t "YOU'RE SET UP! GO WRITE SOME CODE :)" | uv run lolcat
