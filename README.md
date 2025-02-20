# Pupil + RITnet (detect)
<p align="center">
  <img src="https://raw.githubusercontent.com/byeongkang/pupil_with_RITnet/develop/1.png" width="80%" alt="Example workflow image"/>
</p>



<details>
  <summary>한국어 버전</summary>

  ## 소개
  - **프로젝트 개요:**  
    Pupil의 기존 C++ 기반 detect 기능을 RITnet으로 대체하였습니다.

  ## 설치
  - **필수 라이브러리:**  
    - [Pupil](https://github.com/pupil-labs/pupil)에 필요한 라이브러리  
    - [RITnet](https://github.com/AayushKrChaudhary/RITnet)에 필요한 라이브러리  
    - 위 라이브러리들을 사전에 설치한 후 사용하면 됩니다.

  ## 사용법
  - **실행 방법:**  
    인자 값이 이미 설정되어 있으므로, 단순히 `main.py`를 실행하면 자동으로 "capture" 프로세스가 시작됩니다.

  ## 수정 내역
  - **변경 파일:**  
    `pupil_src/shared_moduls/pupil_detector_plugins` 디렉토리 내의  
    - `detector_2d_plugin.py`
    - `detector_base_plugin.py`
    - `pye3d_plugin.py`
  - **주요 변경 사항:**  
    - 위 파일들에 RITnet을 활용한 새로운 메서드 추가 및 기존 메서드 변경이 이루어졌습니다.
    - 부가적인 메서드 및 코드 수정이 포함되어 있습니다.
    - 원래 [DeepVOG](https://github.com/pydsgz/DeepVOG)와 [edgaze](https://github.com/horizon-research/edgaze)를 활용한 detect 기능도 고려되었으나, 최종적으로 RITnet만 사용하였습니다. (차후 코드 정리 예정)

  ## 실행 환경
  - **테스트 환경:**  
    RTX 4090에서 테스트 완료
  - **주의 사항:**  
    코드 실행 시, `eye0`와 `eye1` 카메라의 해상도를 **400 x 400**으로 설정해야 합니다.  
    (192 x 192 해상도에서는 pupil 인식률이 크게 떨어집니다.)

  ## 참고 자료
  - [Pupil by Pupil-labs](https://github.com/pupil-labs/pupil)
  - [RITnet](https://github.com/AayushKrChaudhary/RITnet)
  - [DeepVOG](https://github.com/pydsgz/DeepVOG) (초기 구상)
  - [edgaze](https://github.com/horizon-research/edgaze) (초기 구상)

  ## 라이선스
  - 이 프로젝트는 [Pupil](https://github.com/pupil-labs/pupil) 프로젝트의 일부 코드를 포함하고 있으며, 해당 코드는 **LGPL-3.0** 또는 **GPL-3.0** 라이선스 하에 배포됩니다.
  - 또한, 이 프로젝트는 [RITnet](https://github.com/AayushKrChaudhary/RITnet)의 코드를 포함하고 있으며, 해당 코드는 **MIT License**에 따라 배포됩니다.
  - 본 리포지토리 내에서 수정 및 추가한 코드는 **LGPL-3.0** 라이선스 하에 배포됩니다.
  - 자세한 내용은 프로젝트 루트의 [LICENSE](LICENSE) 파일을 참조하시기 바랍니다.

</details>

<details>
  <summary>English Version</summary>

  ## Overview
  - **Project Summary:**  
    The original C++ based detection functionality of Pupil has been replaced with RITnet.

  ## Installation
  - **Required Libraries:**  
    - Libraries required for [Pupil](https://github.com/pupil-labs/pupil)  
    - Libraries required for [RITnet](https://github.com/AayushKrChaudhary/RITnet)  
    - Please install these dependencies in advance.

  ## Usage
  - **How to Run:**  
    With preset arguments, simply running `main.py` will automatically start the "capture" process.

  ## Modifications
  - **Modified Files:**  
    The following files located in `pupil_src/shared_moduls/pupil_detector_plugins` have been updated:
    - `detector_2d_plugin.py`
    - `detector_base_plugin.py`
    - `pye3d_plugin.py`
  - **Key Changes:**  
    - New methods utilizing RITnet have been added and existing methods modified.
    - Additional code changes and enhancements have been made.
    - Although initial plans included detection methods using [DeepVOG](https://github.com/pydsgz/DeepVOG) and [edgaze](https://github.com/horizon-research/edgaze), ultimately only RITnet was used. (Future code cleanup is planned.)

  ## Environment
  - **Test Environment:**  
    Tested on an RTX 4090.
  - **Important Note:**  
    When running the code, ensure that the camera resolution for both `eye0` and `eye1` is set to **400 x 400**.  
    (A resolution of 192 x 192 significantly reduces the pupil detection rate.)

  ## References
  - [Pupil by Pupil-labs](https://github.com/pupil-labs/pupil)
  - [RITnet](https://github.com/AayushKrChaudhary/RITnet)
  - [DeepVOG](https://github.com/pydsgz/DeepVOG)
  - [edgaze](https://github.com/horizon-research/edgaze)

  ## License
  - This project incorporates portions of the [Pupil](https://github.com/pupil-labs/pupil) project, which is distributed under the **LGPL-3.0** and **GPL-3.0** licenses.
  - Additionally, this project includes code from [RITnet](https://github.com/AayushKrChaudhary/RITnet), which is licensed under the **MIT License**.
  - Modifications and additions made in this repository are distributed under the **LGPL-3.0** license.
  - For further details, please refer to the [LICENSE](LICENSE) file in the root of this repository.

</details>
