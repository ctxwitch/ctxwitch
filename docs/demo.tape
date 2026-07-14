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

Type "witch tour"
Enter
Sleep 3.5s

Type "cd witch-tour"
Enter
Sleep 800ms

# Let the tour make the risky edit for us
Type "witch tour --do"
Enter
Sleep 2s

# The wow moment: behavioral scorecard, not a text diff
Type "witch diff"
Enter
Sleep 6s

# Commit, branch, remove a guardrail — the Breaking verdict
Hide
Type "witch commit -m 'raise refund autonomy' && witch checkout -b risky-change && witch tour --do && clear"
Enter
Sleep 3s
Show

Type "# now remove a safety guardrail and check again..."
Enter
Type "witch diff"
Enter
Sleep 6s

Type "witch eval"
Enter
Sleep 5.5s
