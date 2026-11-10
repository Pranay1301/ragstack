# Inventory and food waste

Inventory tracks ingredients, not dishes. Each menu item maps to a recipe
of ingredient quantities, so every accepted order decrements stock
automatically. When an ingredient falls below its reorder level, the
manager sees it on the dashboard's low-stock list.

Waste is logged against ingredients with a reason: spoiled, overproduction,
or dropped. The dashboard shows the cost of waste over the last fourteen
days, computed from the most recent purchase price of each ingredient.

A stocktake reconciles the book quantity against the counted quantity.
Variances are recorded as adjustments with the counter's initials, so the
audit trail shows who corrected what and when.
