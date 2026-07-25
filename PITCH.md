# Every second counts — full pitch script (~4-5 min)

Delivery notes are in [brackets]. Read the rest as your actual speaking script.

---

## 1. Open with a real moment (30 sec)

"Picture this. It's a Tuesday afternoon. Within about ninety seconds of each other, six
different 911 calls come into dispatch. An elderly man has collapsed downtown. There's a
car crash with injuries across town. A kitchen fire. A bad fall. Somewhere, a dispatcher
has maybe ten seconds to decide: which of our two ambulance crews goes where, in what
order, to get to everyone as fast as possible?

That decision happens, for real, thousands of times a day, in every city in the country.
And most systems still make it the same simple way: send whichever ambulance is closest
to each call, one at a time, as they come in.

[pause] That sounds reasonable. It's actually not optimal. And I'm going to show you why
that gap matters, and how we used quantum computing to try to close it."

## 2. The stakes, with real numbers (45 sec)

"Here's why this isn't just an efficiency problem. According to American Heart
Association response-time research, survival from cardiac arrest drops by roughly 7 to
10 percent for every single minute of delay before help arrives. Not per hour — per
minute.

And the 'closest ambulance' rule that most systems default to has actually been shown,
in published research, to not be the optimal policy. It's just the simplest one. A
simulation of the London Ambulance Service found that a smarter allocation approach was
48 to 54 percent faster on average than the vehicle that was actually historically
dispatched. Other research found optimal deployment could raise the share of
high-priority calls answered within 8 minutes by up to 10 percent.

This isn't a rare edge case either — there are roughly 240 million EMS calls a year in
the US alone. Small, consistent gains here compound into a lot of saved minutes, and a
lot of saved minutes are, statistically, saved lives.

That's the gap. That's what we built Every Second Counts to close."

## 3. Introduce the project (20 sec)

"Every Second Counts is a live dispatch optimizer. You describe incoming emergency calls
in plain English. An AI extracts the real location and severity of each one. We geocode
them onto a real map. And then a quantum computing algorithm — QAOA, running on a real
quantum circuit simulator — decides how to split those calls between two ambulance crews
to minimize total response time.

Let me just show you."

## 4. Live demo — narrate every step (90-120 sec)

[Have the AI intake box already visible. Type or read 5-6 real call descriptions.]

"These are six simultaneous calls — I'm describing them the way a dispatcher actually
would. [click Parse with AI] Right now, a language model is reading each description and
pulling out two things: where it is, and how severe it is. Then we're taking that
location and geocoding it — turning it into a real GPS coordinate — on an actual street
map. [point at map] These aren't fake dots on a blank canvas. That's a real map, real
streets, real distances.

Now here's the part I actually want you to watch closely. [click Run quantum
optimization] What's happening right now is that we've built an actual quantum circuit.
Every single possible way you could split these six calls between two crews — all of
them, all at once — exists simultaneously inside this circuit as something called
superposition. It's not testing one option, then the next, then the next. It's exploring
all of them at the same time.

[point at the collapsing bar chart] Watch this chart. That's not a loading animation.
That is a real, live probability distribution across every possible split, and as the
quantum algorithm runs, it's using quantum interference to boost the good answers and
cancel out the bad ones. Watch it collapse down to a single clear answer.

[once it settles] That's it. That's the answer the quantum computer converged on.
[point at circuit diagram] This is the actual circuit that ran — real qubits, real gates.
[point at Bloch spheres if shown] And these are the individual qubit states — each one
representing a call, moving from undecided to locked in to Crew A or Crew B.

[point at the map] And here's the result laid back onto the real map — red pins are Crew
A, blue pins are Crew B."

## 5. The results, translated into what they mean (30 sec)

"Now look at these three numbers. [point at stats row] This is what a standard greedy
dispatch approach would produce — this is what our quantum optimizer produced instead.
That gap, run through the actual AHA response-time curve, is a real modeled improvement
in survival odds for time-critical calls. We deliberately capped that number so it never
overstates what the research actually supports — we'd rather undersell it than oversell
it."

## 6. The honesty beat — say this with confidence, not apology (25 sec)

"Now I want to tell you something plainly, because I think it actually makes this
project stronger, not weaker: at this problem size — six, eight, ten calls — a classical
computer solves this exact same split instantly. There is no quantum speed advantage
here today, and I'm not going to stand up here and claim there is.

What we built is a mathematically correct, fully working quantum algorithm for a real
problem. We can actually prove that splitting these calls to minimize response time is
mathematically identical to a classic problem called Max-Cut — and Max-Cut is exactly
what QAOA is built to solve. We built it the way it will need to work as quantum
hardware scales to problem sizes — hundreds or thousands of simultaneous decisions —
where classical brute-force search genuinely can't keep up."

## 7. Zoom out and close (25 sec)

"Dispatch is a decision that gets made every single day, in every city, under real time
pressure, with real consequences. We wanted to prove that quantum computing isn't only a
theoretical, someday technology — it can already sit inside a decision like this one,
today, on a simulator, correctly.

Every second really does count. We just tried to make sure the algorithm making that
decision is doing the best job it possibly can. Thank you."

[Stop talking. Let the demo and the numbers sit. Don't rush into Q&A.]

---

## Q&A — rehearse these out loud, don't read them cold

**"Is this actually faster than a classical computer?"**
"Not at this problem size — a classical computer solves this instantly too, and I want
to be upfront about that. The value right now is a correct, working implementation of an
algorithm designed to scale to problem sizes classical brute-force search can't handle."

**"Isn't Max-Cut the most basic quantum computing demo there is?"**
"Yes, QAOA-on-Max-Cut is the standard starting point in quantum optimization — that's
exactly why we didn't stop there. We built the real-world pipeline around it: natural
language call intake, real geocoding, a proven math link between dispatch and Max-Cut,
and an honest, capped impact model, instead of just running the textbook demo and
calling it done."

**"Why should ambulance dispatch even be quantum?"**
"It doesn't have to be, today. But dispatch, routing, and scheduling are exactly the
class of combinatorial optimization problems quantum computing is expected to help with
as hardware scales. We picked a real, high-stakes example from that class instead of an
abstract puzzle."

**"Is the call data real?"**
"The geography is 100% real — real streets, real distances, real geocoding. The specific
calls are simulated because live 911 data isn't publicly accessible. I want to be clear
about that distinction rather than let it look like more than it is."

**"What's the AI actually doing versus the quantum part?"**
"The AI is purely the intake layer — it turns a free-text description into structured
location and severity data. It never touches the optimization. The quantum circuit is
the only thing deciding the actual split."

**"How did you validate the quantum answer is actually correct?"**
"For every scenario, we also run a brute-force classical search that checks every
possible split exhaustively. QAOA lands on the true mathematical optimum — we show that
comparison live, we don't just assert it."

**"What would you build next if you had more time?"**
"Scaling past two crews to N crews, real-time re-optimization as new calls come in
mid-dispatch, and eventually testing on real quantum hardware instead of a simulator to
see how noise affects the result at larger problem sizes."
