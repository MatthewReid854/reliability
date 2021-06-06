.. image:: images/logo.png

-------------------------------------

Copyright information
'''''''''''''''''''''

`reliability` is licenced under the `LGPLv3 <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_.
Some FAQs on the GPL and LGPL are answered `here <https://www.gnu.org/licenses/gpl-faq.html>`_.
The LGPL applies to individuals and businesses with the primary distinction being whether they intend to make a profit through the sale of software that uses `reliability` to perform any function.

Using `reliability` is FREE and unrestricted for both individuals and commercial users.

Individuals and commercial users are FREE to incorporate `reliability` as a part of their own software, subject to the following conditions:

- If linked dynamically (see below):

	- Your software licence is unaffected
	- No credit / attribution is required
	- No changes are permitted (or possible since it is dynamically linked)
	- You must not in any way suggest that the licensor endorses you or your use.
	- You must not sell your software.

- If linked statically (see below):

	- Your software must also be released under the LGPLv3 and you must provide a link to the license.
	- You must give appropriate credit / attribution to the author of `reliability` and the version you have copied.
	- You must indicate any changes that were made.
	- You must not in any way suggest that the licensor endorses you or your use.
	- You must not sell your software.

The major limitation imposed by the LGPL is that individuals and commercial users must not incorporate `reliability` into their own software (either in part or as a whole, either statically or dynamically), and then sell their software.
If you or your company is looking to use `reliability` as part of your software, and you intend to sell your software, then this use falls under a Commercial License and will attract a small fee to permit redistribution.
Please contact alpha.reliability@gmail.com if you believe your usage falls under the Commercial License.

Static vs Dynamic linking
-------------------------

Static linking is where you include part or all of the source code for `reliability` in your package. i.e. you are redistributing source code obtained from `reliability`.
Static linking is strongly discouraged as it does not benefit from improvements and bug fixes made with each release.

Dynamic linking is where your software (typically another Python library or repository) uses `reliability` as a dependancy.
In dynamic linking you are not redistributing any of the source code from `reliability` and are simply using it as an external tool.
Dynamic linking is encouraged as it enables users to use the most recent version of `reliability` which benefits from improvements and bug fixes.

The Licence
-----------

.. image:: images/lgplv3.png

```
GNU LESSER GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright © 2007 Free Software Foundation, Inc. <https://fsf.org/>

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License, supplemented by the additional permissions listed below.

0. Additional Definitions.
As used herein, “this License” refers to version 3 of the GNU Lesser General Public License, and the “GNU GPL” refers to version 3 of the GNU General Public License.

“The Library” refers to a covered work governed by this License, other than an Application or a Combined Work as defined below.

An “Application” is any work that makes use of an interface provided by the Library, but which is not otherwise based on the Library. Defining a subclass of a class defined by the Library is deemed a mode of using an interface provided by the Library.

A “Combined Work” is a work produced by combining or linking an Application with the Library. The particular version of the Library with which the Combined Work was made is also called the “Linked Version”.

The “Minimal Corresponding Source” for a Combined Work means the Corresponding Source for the Combined Work, excluding any source code for portions of the Combined Work that, considered in isolation, are based on the Application, and not on the Linked Version.

The “Corresponding Application Code” for a Combined Work means the object code and/or source code for the Application, including any data and utility programs needed for reproducing the Combined Work from the Application, but excluding the System Libraries of the Combined Work.

1. Exception to Section 3 of the GNU GPL.
You may convey a covered work under sections 3 and 4 of this License without being bound by section 3 of the GNU GPL.

2. Conveying Modified Versions.
If you modify a copy of the Library, and, in your modifications, a facility refers to a function or data to be supplied by an Application that uses the facility (other than as an argument passed when the facility is invoked), then you may convey a copy of the modified version:
a) under this License, provided that you make a good faith effort to ensure that, in the event an Application does not supply the function or data, the facility still operates, and performs whatever part of its purpose remains meaningful, or
b) under the GNU GPL, with none of the additional permissions of this License applicable to that copy.

3. Object Code Incorporating Material from Library Header Files.
The object code form of an Application may incorporate material from a header file that is part of the Library. You may convey such object code under terms of your choice, provided that, if the incorporated material is not limited to numerical parameters, data structure layouts and accessors, or small macros, inline functions and templates (ten or fewer lines in length), you do both of the following:
a) Give prominent notice with each copy of the object code that the Library is used in it and that the Library and its use are covered by this License.
b) Accompany the object code with a copy of the GNU GPL and this license document.

4. Combined Works.
You may convey a Combined Work under terms of your choice that, taken together, effectively do not restrict modification of the portions of the Library contained in the Combined Work and reverse engineering for debugging such modifications, if you also do each of the following:
a) Give prominent notice with each copy of the Combined Work that the Library is used in it and that the Library and its use are covered by this License.
b) Accompany the Combined Work with a copy of the GNU GPL and this license document.
c) For a Combined Work that displays copyright notices during execution, include the copyright notice for the Library among these notices, as well as a reference directing the user to the copies of the GNU GPL and this license document.
d) Do one of the following:
   0) Convey the Minimal Corresponding Source under the terms of this License, and the Corresponding Application Code in a form suitable for, and under terms that permit, the user to recombine or relink the Application with a modified version of the Linked Version to produce a modified Combined Work, in the manner specified by section 6 of the GNU GPL for conveying Corresponding Source.
   1) Use a suitable shared library mechanism for linking with the Library. A suitable mechanism is one that (a) uses at run time a copy of the Library already present on the user's computer system, and (b) will operate properly with a modified version of the Library that is interface-compatible with the Linked Version.
e) Provide Installation Information, but only if you would otherwise be required to provide such information under section 6 of the GNU GPL, and only to the extent that such information is necessary to install and execute a modified version of the Combined Work produced by recombining or relinking the Application with a modified version of the Linked Version. (If you use option 4d0, the Installation Information must accompany the Minimal Corresponding Source and Corresponding Application Code. If you use option 4d1, you must provide the Installation Information in the manner specified by section 6 of the GNU GPL for conveying Corresponding Source.)

5. Combined Libraries.
You may place library facilities that are a work based on the Library side by side in a single library together with other library facilities that are not Applications and are not covered by this License, and convey such a combined library under terms of your choice, if you do both of the following:
a) Accompany the combined library with a copy of the same work based on the Library, uncombined with any other library facilities, conveyed under the terms of this License.
b) Give prominent notice with the combined library that part of it is a work based on the Library, and explaining where to find the accompanying uncombined form of the same work.

6. Revised Versions of the GNU Lesser General Public License.
The Free Software Foundation may publish revised and/or new versions of the GNU Lesser General Public License from time to time. Such new versions will be similar in spirit to the present version, but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number. If the Library as you received it specifies that a certain numbered version of the GNU Lesser General Public License “or any later version” applies to it, you have the option of following the terms and conditions either of that published version or of any later version published by the Free Software Foundation. If the Library as you received it does not specify a version number of the GNU Lesser General Public License, you may choose any version of the GNU Lesser General Public License ever published by the Free Software Foundation.

If the Library as you received it specifies that a proxy can decide whether future versions of the GNU Lesser General Public License shall apply, that proxy's public statement of acceptance of any version is permanent authorization for you to choose that version for the Library.
```

Reference: `https://www.gnu.org/licenses/lgpl-3.0.html <https://www.gnu.org/licenses/lgpl-3.0.html>`_