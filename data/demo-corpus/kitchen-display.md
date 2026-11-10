# Kitchen display system

The kitchen display system, or KDS, replaces printed tickets. Every new
order becomes a ticket on the station screen. Routing follows the menu
item's station: drinks go to the bar, breads go to the tandoor, and
curries go to the hot line. A ticket appears on every station that owns at
least one of its items.

Tickets move through four states: new, accepted, preparing, and ready.
Cooks tap a ticket to accept it, tap again when they start preparing, and
mark it ready when the food is plated. A new ticket plays a short sound so
staff notice it during a loud service. The counter screen shows the same
states, so the front of house can answer "where is my order" without
walking to the kitchen.

If a ticket stays in the new state for more than five minutes it turns
amber, and after ten minutes it turns red. These thresholds are per-station
settings.
