.. image:: images/logo.png

-------------------------------------

What is Accelerated Life Testing
''''''''''''''''''''''''''''''''

Accelerated life testing (ALT) is a method of test and analysis to determine how failures would likely occur in the future. ALT is a popular method of testing because of its ability to "speed up time". ALT is often used when we can not afford to wait for failures to occur at their normal rate but we need to know how failures are likely to occur in the future.

Consider an electronics manufacturer who wants to know how many failures will occur in 10 years (possibly for warranty purposes). If the component being tested has a mean life of 30 years, the manufacturer cannot reasonably spend several years performing a reliability test as they are ready to release their product on the market soon. By increasing the stress on the component, failure will be induced more rapidly. Done correctly, this is equivalent to speeding up the passage of time. The electronics manufacturer can collect failure data at a variety of stresses, fit the appropriate life-stress model, and then enter the "use stress" into the life-stress model to determine the failure distribution that is expected to occur at the use stress.

ALT testing is also a very useful way to determine the effectiveness of derating. Derating is the process of reducing the load (typically voltage or current) on a component below its "rated" load, or equivlently selecting a component that is rated above the design load. How much the component life will be extended can be quantitatively measured using an ALT test to find the life-stress model.

To ensure the ALT test is performed correctly, the analyst must ensure that the failure modes are the same at each stress. This will be evidenced by the shape parameter of the distribution as a changing shape parameter will show the failure mode is changing, though it is desirable that each failed component be examined to ensure that the failure mode being studied was the failure mode experienced. As with any model fitting the analyst must ensure there is sufficient data to fit the model such that the results are meaningful. This means the ALT test needs sufficient stresses (usually 3 or more) and sufficient failures (as many as you can afford to test) at each stress.

ALT tests may either be single stress or dual stress. In dual stress models, there are two stresses being tested, such as temerature and humidity. The testing process is largely the same though users should note that with an additional variable in the model it is highly desirable to have more failure data to fit the model accurately. Additionally, it is important that both stresses are varied sufficiently (relative to the design load) so that the life-stress curve (or life-stress surface in the case of dual stress models) has enough data over enough range to be fitted accurately.

Types of ALT
""""""""""""

The way an ALT test is performed depends on the stress profile. There are two popular methods to perform an ALT test; using a constant stress profile, and using a step stress profile. In a constant stress profile, each item under test only ever experiences a single stress level. In a step stress profile each item begins at the lowest stress which is held for a period of time before being stepped up to higher and higher levels. Constant stress profiles are mathematically easier to fit and understand and therefore are more popular. Step stress profiles are useful when you only have a limited number of items and you do not know at what stress you should test them. Selecting a stress that is too low may result in no failures so the opportunity to use the same components (which have not yet failed) from the first test in subsequent tests at higher levels is advantageous. 

.. image:: images/ALT_stress_profiles.png

Within `reliability` there are 24 constant stress ALT models currently implemented (12 single stress and 12 dual stress). Step stress models are not yet implemented within `reliability` though this feature is planned for a future release. Users seeking to fit a step stress profile may want to consider using Reliasoft's `ALTA <http://reliawiki.com/index.php/Time-Varying_Stress_Models>`_.

The mathematical formulation of ALT models is explained further in the section on `Equations of ALT models <https://reliability.readthedocs.io/en/latest/Equations%20of%20ALT%20models.html>`_.

ALT vs HALT vs ESS vs Burn-in
"""""""""""""""""""""""""""""

Highly Accelerated Life Testing (HALT) is a type of testing to determine how things fail, rather than when things will fail. HALT has no survivors as the primary goal is to record the way in which items fail (their failure mode) so that design improvements can be made to make the design more resistant to those failure modes. HALT is mostly qualitative while ALT is quantitative. Since HALT is qualitative, there are no models required for fitting failure data.

Environmental Stress Screening (ESS) is a process of exposing components to a series of stresses which they are likely to experience throughout their lifetime such as rapid thermal cycling, vibration, and shock loads. These stresses precipitate latent manufacturing defects as early failures. ESS is often confused with burn-in since both are a screening process to remove weak items from a batch, effectively removing the infant mortality failures from the customer's experience. Unlike burn-in, ESS uses a range of loads, more than just thermal and voltage as is seen in burn-in. ESS does not simulate the component's design environment or usage profile, though it should use a range of stresses (or combinations of stresses) which are on the upper or lower limits of the component's design limit. It is important that the applied stress does not approach the mechanical, electrical, or thermal stress limits of any component as ESS is not intended to cause damage or fatigue. Ideally, components that pass ESS will not have had any of their life consumed during the ESS process. Each screening profile must be tailored specifically for the component/product on which it is applied.

Burn-in involves stressing components with a higher load than their design load such that the "weak" items are screened out through failure. Often confused with ESS, burn-in can be though of as a subset of ESS with a focus on thermal or electrical loads generally used for screening electrical components. The strength of a population of components will always have some variability (that can be modeled using a probability distribution). By "burning-in" the population of components, manufacturers can screen out (through failure) the lower part of the distribution (of strengths) to be left with only the stronger components from the batch. Burn-in is only appropriate for stresses which cause wear-in or random failure modes (not wear out failure modes which accumulate damage). If the stress causes cumulative damage then the burn-in process would consume some of the component's life. MIL-STD-883C defines a burn-in test as `Burn-in is a test performed for the purpose of screening or eliminating marginal devices, those with inherent defects or defects resulting from manufacturing aberrations which cause time and stress dependent failures.`

Further reading
"""""""""""""""

Reliasoft's `Accelerated Life Testing Data Analysis Reference <http://reliawiki.com/index.php/Accelerated_Life_Testing_Data_Analysis_Reference>`_ provides a great deal more information on ALT.
