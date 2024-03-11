---
marp: true
title: Test
theme: default #gaia #uncover
class: #invert
math: mathjax
---

![bg 90% right](zstack1.png)

# Anticipating Escalation
**Actionable Insights with Actor-Embeddings and Transformers in Conflict Forecasting**

&nbsp;
&nbsp;
w/ Mihai Croicu

---

![bg 90% right](zstack1.png)

### Very preliminary - basically a pitch...

---

![bg 100% left:22%](east.png)

**VIEWS** already produces innovative and pioneering conflict forecasting

We have a robust **network of partners** who also work on conflict forecasting

Practitioners are experimenting with their own **early warning systems**

---
![bg 100%  right:22%](west2.png)

But most, if not all, current approaches have one **detrimental feature** in common: **inertia**.

---

<img src="timelapse.gif" alt="My cool gif">


---
![bg 100% left:22%](east.png)


**Seth Caldwell,** 

Data Scientist at UN OCHA Centre for Humanitarian Data and a member of the scoring committee for the VIEWS prediction competitions

Has thoughts on this:

---
![bg 100%  right:22%](west2.png)

**The thoughts:** 

>"[There is] insufficient justification for exclusively relying on conflict prediction models to drive anticipatory action due to several factors:"

- Poor performance in predicting the onset of new conflicts.
- The lack of clear connection between predicted conflict and resulting humanitarian impact.
- The dominance of ongoing conflict as a predictor of future conflict. 

&nbsp;
[Source](https://centre.humdata.org/assessing-the-technical-feasibility-of-conflict-prediction-for-anticipatory-action/)

---
![bg 100% left:22%](east.png)

And indeed are **challenges**:

- **Conflict history**, in some form, is the primary predictor in most current systems.

- Socio-economic, institutional, demographic, and geographic features are usually aslo included in models but **lack varience in signals** due to inherent inertia or granularity of measurment.


---
![bg 100%  right:22%](west2.png)


Among a number of **recommendations**, Seth and his team rightly note that, for our models to be useful for anticipatory action in the humanitarian sector, we should:


>**"Focus models on predicting shifts in conflicts, such as an increase in intensity or onset."**

&nbsp;
[Source](https://centre.humdata.org/assessing-the-technical-feasibility-of-conflict-prediction-for-anticipatory-action/)

---
![bg 100% left:22%](east.png)


**Fair is fair**, and we do hold Seth in high regard.
&nbsp;
So now we are creating a **new model** for Seth.
&nbsp;
A model specifically designed to **generate actionable insights** by utilizing actor-embeddings and transformer networks to forecast **escalatory patterns** in violent conflict.

---
![bg 100%  right:22%](west2.png)

&nbsp;
When desgining this new model, we will emphasize **escalatory patterns**. 
&nbsp;
The fact that we are **not exclusively focusing on onsets is deliberate**.
&nbsp;
Forecasting onsets is considered the holy grail by some conflict scholars, **but...**

---
![bg 100% left:22%](east.png)

Mali, Chad, Guinea, Burkina Faso, Niger... **Limited** room for action.
&nbsp;
Acting on anticipated escalations in conflict zones where humanitarians and international actors are **already present and mandated** is likely more realistic.
&nbsp;
But **"both is good"**, so we will try to achieve both.

---
![bg 100%  right:22%](west2.png)
**In a nutshell:** 

:mag_right:  The unit of prediction will be **armed actors** (groups).
&nbsp;
:dart: The prediction target will be the number of future battle-related **fatalities** produced by a given actor each month.
&nbsp;
:chart_with_upwards_trend: We will forecast a **sequence** of 12 months.
&nbsp;
:robot: To create these forecasts, we will use **transformer neural networks**.
&nbsp;
:newspaper: And as input for these transformers, we will create **actor embeddings** using text data (news sources, wikis, etc.).

---
![bg 100% left:22%](east.png)

Lets unpack some of this jargon, specifcally 
&nbsp;
- **Embeddings**
&nbsp;
- **Transformers** 
&nbsp;
- **Sequence**


---
![bg 100%  right:22%](west2.png)

**Embeddings (1):**

Embeddings act as **multidimensional summaries** of information. For instance, consider a 2D actor space with a capability and a ideology dimensions.
&nbsp;

Now, imagine having an **actor space** with 1000 dimensions for a much more nuanced summary.


---
![bg 100% left:22%](east.png)


**Embeddings (2):**

Machine learning models are trained using large amount of **text data** to position armed groups (relative to each other) in a highly multidimensional space.
&nbsp;

Now, Actors are no longer just names or numbers. They are represented by vectors in this multidimensional space - but their **positions can change over time**.



---
![bg 100%  right:22%](west2.png)

**Transformers/Attention (1)**

Transformers take the multidimensional **actor embeddings as inputs**.
&nbsp;

Transformers employ an attention mechanism to learn **relationships and similarities** between the input embeddings - which here represents different actors at different times.


---
![bg 100% left:22%](east.png)


**Transformers/Attention (2)**

When "asked" to review a given actor, the transformer compares it to itself and to **all other observed actors** across time.
&nbsp;

By discerning these patterns, the model constructs the **most likely future pattern for the actor** under review, (hopefully) enabling the model to anticipate violent actions.

---
![bg 100%  right:22%](west2.png)
**Sequence**
&nbsp;

Notably, the target and thus the output will be **a sequence of months**.
&nbsp;
Current research suggests that the **stepshifter** models and **autoregressive** models struggle to discern between temporal units.
&nbsp;
I.e., the probability mass is spread across time. **Forecasting a full sequence** at each step might alleviate this." 

---
![bg 100% left:22%](east.png)
**Additional features (1)**

Beyond the goal of achieving accurate predictions, this approach helps in incorporating new actors into prediction frameworks.
&nbsp;
I.e, it is challenging to assess whether a new actor will exhibit behaviors similar to one or more known actors and indeed to which known actors. 
&nbsp;
Positioning new actors within an actor space, which can be continually updated, serves as a valuable (hierarchical-ish) prior. 

---
![bg 100%  right:22%](west2.png)
**Additional features (2)**

Furthermore, examining the relationships among established actors in embedding space might be a worthwhile study in itself.

I.e. understanding how these known actors relate to each other in this multidimensional space is likely of substantial interest.

---

**Mihai Croicu &** 
**Simon Polichinel von der Maase**

&nbsp;
&nbsp;
:mailbox: simmaa@prio.org
:octopus: https://github.com/Polichinel
:european_castle: [PRIO, Oslo](https://www.prio.org/)

