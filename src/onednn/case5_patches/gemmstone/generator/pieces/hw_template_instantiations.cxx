/*******************************************************************************
 * Case5 gemmstone Generator instantiations (Gen12LP + XeHPG + Xe2).
 *******************************************************************************/
#ifdef _MSC_VER
#pragma warning(disable : 4661)
#endif

template class Generator<HW::Gen12LP>;
template class Generator<HW::XeHPG>;
template class Generator<HW::Xe2>;
