# VHS recording script for the README demo GIF.
# Render with:  vhs docs/demo.tape   (from repo root)
# Output: docs/demo.gif — embed at the top of README.md.

Output docs/demo.gif

Set FontSize 15
Set Width 1100
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 12

# Work in a scratch dir so the repo stays clean
Hide
Type "export PATH=$HOME/Library/Python/3.9/bin:$PATH && cd $(mktemp -d) && clear"
Enter
Sleep 500ms
Show

# Start the tour — it offers two ways in (scan an existing agent, or witch.yaml)
Type "witch tour"
Enter
Sleep 4s

Type "cd witch-tour"
Enter
Sleep 800ms

# Path A — scan an existing agent, no rewrite required
Type "# already have an agent? scan it — no rewrite, no witch.yaml"
Enter
Type "witch scan agent.py"
Enter
Sleep 4.5s

# Make a sample change: drop the investment-advice guardrail (the tour does it)
Type "witch tour --edit-agent"
Enter
Sleep 2s

# The payoff: a behavioral scorecard straight from the code diff — Breaking.
Type "witch scan agent.py --diff HEAD"
Enter
Sleep 6.5s
