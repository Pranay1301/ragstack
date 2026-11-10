# Shift brief

The shift brief is a deterministic pre-shift summary generated from the
point of sale data, with no AI model involved. It reads the last fourteen
days of orders and answers fixed questions: which items sold most, which
hours were busiest, what the average bill value was, and how the coming
shift's staffing compares to last week.

Because the brief is rule-based, the same data always produces the same
brief. That is a deliberate choice: a manager can check a number against
the underlying orders and trace exactly how it was computed. The brief
shows its working - every figure links to the orders behind it.

The brief does not forecast. It reports what happened and flags unusual
patterns, such as an item selling twice its median daily count, but it
does not predict tomorrow's sales.
