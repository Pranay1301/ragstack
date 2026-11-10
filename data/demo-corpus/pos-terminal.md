# Point of sale terminal

The point of sale terminal is where staff ring up orders at the counter.
Each order line carries an item, a quantity, and any modifiers such as
"extra spicy" or "no onion". Taxes are applied automatically from the
restaurant's configured rates: 5 percent on most food items and 18 percent
on packaged goods in this demo configuration.

The terminal works offline. When the internet drops, orders are queued in
local storage with their timestamps, and they sync to the back office when
connectivity returns. A conflict - the same bill edited in two places - is
resolved in favour of the edit with the later timestamp.

Payments are recorded against the bill: cash, card, or UPI. A bill is
closed only when the recorded payments add up to its total. Splitting a
bill across guests is supported; each split part gets its own payment row.
