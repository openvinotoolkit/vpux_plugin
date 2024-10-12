# How to setup windows environment

This document introduces how to set up windows environment for Driver Compiler package building step by step.

## Contents:
* Step 1: [Install CMake](#Install_CMake)
* Step 2: [Install Git](#Install_Git)
* Step 3: [Install Python](#Install_Python)
* Step 4: [Install Visual Studio 2019 Community](#Install_Visual_Studio_2019_Community)
* Step 5: [Install Windows SDK](#Install_Windows_SDK)
* Step 6: [Install Windows WDK](#Install_Windows_WDK)

> Notice: Please do not forget set their environment variables in system PATH, if needed.

## <span id="Install_CMake">Step 1: Install CMake</span>

1)	According to [openvino/install_build_dependencie.sh](https://github.com/openvinotoolkit/openvino/blob/master/install_build_dependencies.sh#L208), the cmake minor version is ‘3.20.0’. Here choose the cmake (version 3.24.0.rc1) in the following link to use. Use the [link](https://github.com/Kitware/CMake/releases/download/v3.24.0-rc1/cmake-3.24.0-rc1-windows-x86_64.msi) to download and turn to the download path.

2) Double click the `.msi` file to start installation, click `Next`.
 
3) Accept the license and click `Next`.
 
4) Choose Add CMake to the system PATH for all users and click `Next`.
 
5) Click `Next` to set the default cmake install path.
 
6) Click `Install`, wait for the installation.
 
7) Click `Finish`.

## <span id="Install_Git">Step 2: Install Git</span>

1) Download git for Windows with the [link](https://github.com/git-for-windows/git/releases/download/v2.36.1.windows.1/Git-2.36.1-64-bit.exe)

2) Double click the .exe file to install, click `Yes` to allow the installation.
 
3) Accept the license and click `Next`.

4) Click `Next` and set the default git install path.

5) Check these components and click `Next`.

	<img src=./imgs/git_components.png width=60% />

6) Click `Next` with default option until install.

7) Click `Install` and Wait for installing.
 
8) Click `Finish`.


## <span id="Install_Python">Step 3: Install Python</span>

1) According to OV/install_build_dependencie.sh and how-to-build.md, the minor python version is `3.9`. Here choose the Python 3.9.10 to use. Download with the [link](https://www.python.org/ftp/python/3.9.10/python-3.9.10-amd64.exe) 

2) Double click the .exe file to install Python.

3) Check Add Python 3.9 to PATH, and click `Install Now`.
 
4) Wait for the installation.
 
5) Click `Disable path length limit`, click `Yes` for the pop out window.

6) Then click `Close` to finish installation for Python.

 
## <span id="Install_Visual_Studio_2019_Community">Step 4: Install Visual Studio 2019 Community</span>

Below is an example for Visual Studio2019.

1) According to [OpenVINO Project]/install_build_dependencie.sh, the Visual Studio needed to use is 2019 with version 16.3 or higher. Here choose Visual Studio 2019 community version 16.11 in the [link](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads). Click the `Download` and then turn to the download path.

	<img src=./imgs/VisualStudioDownloadweb.png width=80% />

2) Double click vs_community__xxxxxxxxx.exe to start install.
 
3) Check the following components `desktop development with c++` and all opetion before `JavaScript diagnostics`.
	
	<img src=./imgs/VisualStudiocomponents1.png width=90% />

	To avoid some build issue, we need to install `spectre mitigated libs` as follow step. Go to the `Individual componets` part and search for `spectre mitigated libs`. Then check the corresponding box and click `install while downloading`.

	<img src=./imgs/VisualStudiocomponents2.png width=90% />
	
4) Wait for the installation and then Visual Studio 2019 Community is installed successfully.
 
5) If you want to build with ninja, please Open Visual Studio Installer, choose the `Modify` and then `Individual components`. Next, search for C++ CMake Tools for Windows and check the box, then click the install in right part and waiting for install. If you can not install by thish way, please refer to the part of "Getting Ninja" section on https://ninja-build.org/.


## <span id="Install_Windows_SDK">Step 5: Install Windows SDK</span>

You can download the windows SDK with the [link](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/). You can choose the one suitable for your windows environment. 

1) Here choose the 10.0.22000.194 based on our current system environment, click `Install SDK`.

	<img src=./imgs/SDKdownloadweb.png width=90% />

2) After finishing downloading winsdksetup.exe, turn to the download path. Then, double click `winsdks

3) Choose `No` when collect insight for the windows Kits and Click `Next`.

4) Click `Next` to accept the license.

5) Check box for all. Notice: here has installed Application Verifier and WinDbg (windbg also can be installed by this link). Then click `Install`.

	<img src=./imgs/SDKcomponents.png width=60% />

6) Wait for the installation and the click `Close` to finish installation.


## <span id="Install_Windows_WDK">Step 6: Install Windows WDK</span>

You can download the windows WDK with the [guide link](https://learn.microsoft.com/en-ie/windows-hardware/drivers/other-wdk-downloads). Go to the `step2: Install the WDK`(or just use this [link](https://learn.microsoft.com/en-us/windows-hardware/drivers/other-wdk-downloads?source=recommendations#step-2-install-the-wdk)) in [guide link](https://learn.microsoft.com/en-ie/windows-hardware/drivers/other-wdk-downloads). Choose a one suitable one for your windows environment to downloading. 

1) Here choose the windows 11 with version 21H22 based on current system environment, click `Windows 11, version 21H2 WDK`.

	<img src=./imgs/WDKdownloadweb.png width=60% />
	 
2) Double click wdksetup.exe, and click Next with default installer path.
 
3) Choose `No` when collect insight for the windows Kits and Click `Next`.
 
4) Click Next and click `Install`.

5) Then maybe get a message box as follow. Choose the `Install Windows Driver Kit Visual Studio extension` and click the `close` button.

	<img src=./imgs/WDKinstalled.png width=60% />

6) Prompt a windows to info you the change of installation, click `yes` and trun to install page and click `install`.

	<img src=./imgs/WDKextension.png width=60% />

6) Then, waiting for the installation and the final installation result is as follows and click `Close`.

	<img src=./imgs/WDKextensioninstalled.png width=60% />

Note: Please note that api validator tool is part of WDK for Windows 10. If you did not special the install path in the beginning of installation WDK, this tool is located in C:\Program Files (x86)\Windows Kits\10\bin. Otherwise, it will be in the path you specified.

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino