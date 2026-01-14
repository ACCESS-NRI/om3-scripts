!==============================================================================
! ESMF_routehandle_offline_gen.f90 - offline route handle generation for ACCESS-OM3
! ==============================================================================
! Precomputes ESMF RouteHandles offline, write .RH files so the mediator can just 
! load them at runtime.

! how to use:
! mpirun -n <totalPets> ./ESMF_route_handle_offline_generation \
! --mesh_atm <atm_mesh_file> \
! --mesh_ice <ice_mesh_file> \
! --mesh_ocn <ocn_mesh_file> \
! --mesh_rof <rof_mesh_file> \
!
! Contact:
!   Minghang Li <minghang.li1@anu.edu.au>
!==============================================================================

program ESMF_routehandle_offline_gen

    use ESMF
    implicit none

    character(len=512) :: mesh_atm_path, mesh_ice_path, mesh_ocn_path, mesh_rof_path
    integer :: rc ! return code for ESMF

    type(ESMF_VM) :: vm ! ESMF Virtual Machine
    integer :: petCount, petLocal ! total ranks and local rank

    type(ESMF_Mesh) :: mesh_atm, mesh_ice, mesh_ocn, mesh_rof ! esmf meshes
    type(ESMF_Field) :: field_atm, field_ice, field_ocn, field_rof ! node-based esmf fields - for bilinear and patch regridding
    type(ESMF_Field) :: field_atm_elem, field_ice_elem, field_ocn_elem ! element-based fields - for conservative regridding

    type(ESMF_RouteHandle) :: rh_a2o_patch, rh_a2o_bilnr, rh_a2o_consf ! atm -> ocn route handles
    type(ESMF_RouteHandle) :: rh_a2i_patch, rh_a2i_bilnr, rh_a2i_consf ! atm -> ice route handles
    type(ESMF_RouteHandle) :: rh_o2i_redist, rh_i2o_redist ! ocn <-> ice route handles

    ! Below two are constants: https://github.com/ESCOMP/CMEPS/blob/18eb93be8f80286739fbea7b500a069fb0d71aa8/mediator/med_constants_mod.F90#L17
    integer(ESMF_KIND_I4), parameter :: srcMaskVal = -987987_ESMF_KIND_I4 
    integer(ESMF_KIND_I4), parameter :: dstMaskVal = 0_ESMF_KIND_I4

    call parse_input(mesh_atm_path, mesh_ice_path, mesh_ocn_path, mesh_rof_path) ! read input arguments

    call ESMF_Initialize(logKindFlag=ESMF_LOGKIND_MULTI, rc=rc) ! init ESMF
    call chk('ESMF_Initialize', rc)

    call ESMF_VMGetGlobal(vm, rc=rc)
    call chk('ESMF_VMGetGlobal', rc)

    call ESMF_VMGet(vm, petCount=petCount, localPet=petLocal, rc=rc)
    call chk('ESMF_VMGet', rc)

    ! Load meshes
    call MeshFromFile(mesh_atm_path, mesh_atm, rc)
    call chk('MeshFromFile atm', rc)

    call MeshFromFile(mesh_ice_path, mesh_ice, rc)
    call chk('MeshFromFile ice', rc)

    call MeshFromFile(mesh_ocn_path, mesh_ocn, rc)
    call chk('MeshFromFile ocn', rc)

    call MeshFromFile(mesh_rof_path, mesh_rof, rc)
    call chk('MeshFromFile rof', rc)

    ! Create node-based fields on meshes for bilinear and patch regridding
    field_atm = ESMF_FieldCreate(mesh=mesh_atm, meshloc=ESMF_MESHLOC_NODE, &
    typekind=ESMF_TYPEKIND_R8, name='field_atm', rc=rc)
    call chk('ESMF_FieldCreate field_atm', rc)

    field_ice = ESMF_FieldCreate(mesh=mesh_ice, meshloc=ESMF_MESHLOC_NODE, &
    typekind=ESMF_TYPEKIND_R8, name='field_ice', rc=rc)
    call chk('ESMF_FieldCreate field_ice', rc)

    field_ocn = ESMF_FieldCreate(mesh=mesh_ocn, meshloc=ESMF_MESHLOC_NODE, &
    typekind=ESMF_TYPEKIND_R8, name='field_ocn', rc=rc)
    call chk('ESMF_FieldCreate field_ocn', rc)

    field_rof = ESMF_FieldCreate(mesh=mesh_rof, meshloc=ESMF_MESHLOC_NODE, &
    typekind=ESMF_TYPEKIND_R8, name='field_rof', rc=rc)
    call chk('ESMF_FieldCreate field_rof', rc)

    ! Create element-based fields on meshes for conservative regridding
    field_atm_elem = ESMF_FieldCreate(mesh=mesh_atm, meshloc=ESMF_MESHLOC_ELEMENT, &
    typekind=ESMF_TYPEKIND_R8, name='field_atm_elem', rc=rc)
    call chk('ESMF_FieldCreate field_atm(elem)', rc)

    field_ice_elem = ESMF_FieldCreate(mesh=mesh_ice, meshloc=ESMF_MESHLOC_ELEMENT, &
    typekind=ESMF_TYPEKIND_R8, name='field_ice_elem', rc=rc)
    call chk('ESMF_FieldCreate field_ice(elem)', rc)

    field_ocn_elem = ESMF_FieldCreate(mesh=mesh_ocn, meshloc=ESMF_MESHLOC_ELEMENT, &
    typekind=ESMF_TYPEKIND_R8, name='field_ocn_elem', rc=rc)
    call chk('ESMF_FieldCreate field_ocn(elem)', rc)

    ! Now build the RHs
    ! With masking https://github.com/esmf-org/esmf/blob/c8ecfef18ee14c18bb15d6b5c60a7ecc27ecd206/src/Infrastructure/Field/examples/ESMF_FieldRegridMaskEx.F90#L312
    ! atm -> ocn patch
    call ESMF_FieldRegridStore(srcField=field_atm, srcMaskValues=(/ srcMaskVal /), &
    dstField=field_ocn, dstMaskValues=(/ dstMaskVal /), &
    unmappedaction=ESMF_UNMAPPEDACTION_IGNORE, &
    routehandle=rh_a2o_patch, &
    regridMethod=ESMF_REGRIDMETHOD_PATCH, rc=rc)
    call chk('ESMF_FieldRegridStore atm->ocn patch', rc)

    ! atm -> ocn bilinear
    call ESMF_FieldRegridStore(srcField=field_atm, srcMaskValues=(/ srcMaskVal /), &
    dstField=field_ocn, dstMaskValues=(/ dstMaskVal /), &
    unmappedaction=ESMF_UNMAPPEDACTION_IGNORE, &
    routehandle=rh_a2o_bilnr, &
    regridMethod=ESMF_REGRIDMETHOD_BILINEAR, &
    rc=rc)
    call chk('ESMF_FieldRegridStore atm->ocn bilinear', rc)

    ! atm -> ocn conservative
    call ESMF_FieldRegridStore(srcField=field_atm_elem, srcMaskValues=(/ srcMaskVal /), &
    dstField=field_ocn_elem, dstMaskValues=(/ dstMaskVal /), &
    unmappedaction=ESMF_UNMAPPEDACTION_IGNORE, &
    routehandle=rh_a2o_consf, &
    regridMethod=ESMF_REGRIDMETHOD_CONSERVE, &
    rc=rc)
    call chk('ESMF_FieldRegridStore atm->ocn consf', rc)

    ! atm -> ice patch
    call ESMF_FieldRegridStore(srcField=field_atm, srcMaskValues=(/ srcMaskVal /), &
        dstField=field_ice, dstMaskValues=(/ dstMaskVal /), &
        unmappedaction=ESMF_UNMAPPEDACTION_IGNORE, &
        routehandle=rh_a2i_patch, &
        regridMethod=ESMF_REGRIDMETHOD_PATCH, &
        rc=rc)
    call chk('ESMF_FieldRegridStore atm->ice patch', rc)

    ! atm -> ice bilinear
    call ESMF_FieldRegridStore(srcField=field_atm, srcMaskValues=(/ srcMaskVal /), &
        dstField=field_ice, dstMaskValues=(/ dstMaskVal /), &
        unmappedaction=ESMF_UNMAPPEDACTION_IGNORE, &
        routehandle=rh_a2i_bilnr, &
        regridMethod=ESMF_REGRIDMETHOD_BILINEAR, &
        rc=rc)
    call chk('ESMF_FieldRegridStore atm->ice bilinear', rc)

    ! atm -> ice conservative
    call ESMF_FieldRegridStore(srcField=field_atm_elem, srcMaskValues=(/ srcMaskVal /), &
    dstField=field_ice_elem, dstMaskValues=(/ dstMaskVal /), &
    unmappedaction=ESMF_UNMAPPEDACTION_IGNORE, &
    routehandle=rh_a2i_consf, &
    regridMethod=ESMF_REGRIDMETHOD_CONSERVE, &
    rc=rc)
    call chk('ESMF_FieldRegridStore atm->ice consf', rc)

    ! ocn -> ice redist
    call ESMF_FieldRedistStore(srcField=field_ocn, &
        dstField=field_ice, &
        routehandle=rh_o2i_redist, &
        rc=rc)
    call chk('ESMF_FieldRedistStore ocn->ice', rc)

    ! ice -> ocn redist
    call ESMF_FieldRedistStore(srcField=field_ice, &
        dstField=field_ocn, &
        routehandle=rh_i2o_redist, &
        rc=rc)
    call chk('ESMF_FieldRedistStore ice->ocn', rc)

    ! save all RHs to files
    call ESMF_RouteHandleWrite(rh_a2o_patch, fileName='rh_a2o_patch.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_a2o_patch', rc)
    call ESMF_RouteHandleWrite(rh_a2o_bilnr, fileName='rh_a2o_bilnr.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_a2o_bilnr', rc)
    call ESMF_RouteHandleWrite(rh_a2o_consf, fileName='rh_a2o_consf.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_a2o_consf', rc)
    call ESMF_RouteHandleWrite(rh_a2i_patch, fileName='rh_a2i_patch.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_a2i_patch', rc)
    call ESMF_RouteHandleWrite(rh_a2i_bilnr, fileName='rh_a2i_bilnr.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_a2i_bilnr', rc)
    call ESMF_RouteHandleWrite(rh_a2i_consf, fileName='rh_a2i_consf.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_a2i_consf', rc)
    call ESMF_RouteHandleWrite(rh_o2i_redist, fileName='rh_o2i_redist.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_o2i_redist', rc)
    call ESMF_RouteHandleWrite(rh_i2o_redist, fileName='rh_i2o_redist.RH', rc=rc)
    call chk('ESMF_RouteHandleWrite rh_i2o_redist', rc)

    call ESMF_Finalize(rc=rc)
    call chk('ESMF_Finalize', rc)

contains

    subroutine parse_input(mesh_atm, mesh_ice, mesh_ocn, mesh_rof)
        character(len=512), intent(out) :: mesh_atm, mesh_ice, mesh_ocn, mesh_rof
        integer :: i, n
        character(len=512) :: a

        mesh_atm = ''
        mesh_ice = ''
        mesh_ocn = ''
        mesh_rof = ''

        n = command_argument_count() ! loop over argv
        i = 1

        do while (i <= n)
            call get_command_argument(i, a)
            select case (trim(a))
            case ('--mesh_atm')
                i = i + 1
                call get_command_argument(i, mesh_atm)
            case ('--mesh_ice')
                i = i + 1
                call get_command_argument(i, mesh_ice)
            case ('--mesh_ocn')
                i = i + 1
                call get_command_argument(i, mesh_ocn)
            case ('--mesh_rof')
                i = i + 1
                call get_command_argument(i, mesh_rof)
            end select
            i = i + 1
        end do
        if (len_trim(mesh_atm) == 0 .or. & 
            len_trim(mesh_ice) == 0 .or. &
            len_trim(mesh_ocn) == 0 .or. &
            len_trim(mesh_rof) == 0) then
            write(*,*) 'Error: Missing required arguments.'
            write(*,*) 'Usage: ESMF_routehandle_offline_gen --mesh_atm <atm_mesh_file> --mesh_ice <ice_mesh_file> --mesh_ocn <ocn_mesh_file> --mesh_rof <rof_mesh_file>'
            stop 1
        end if
    end subroutine parse_input

    subroutine chk(routine, rc)
        character(len=*), intent(in) :: routine
        integer, intent(in) :: rc
        if (rc /= ESMF_SUCCESS) then
            write(*,*) 'Error in ', trim(routine), ', rc = ', rc
            call ESMF_Finalize(endflag=ESMF_END_ABORT)
            stop 2
        end if
    end subroutine chk

    subroutine MeshFromFile(meshFile, mesh, rc)
    use ESMF
    implicit none
    character(len=*), intent(in)  :: meshFile
    type(ESMF_Mesh),  intent(out) :: mesh
    integer,          intent(out) :: rc

    rc = ESMF_SUCCESS

    mesh = ESMF_MeshCreate(trim(meshFile), ESMF_FILEFORMAT_ESMFMESH, rc=rc)

    if (rc /= ESMF_SUCCESS) then
        write(*,*) 'MeshCreate(from file) failed for ', trim(meshFile), ' rc=', rc
    end if
    end subroutine MeshFromFile

end program ESMF_routehandle_offline_gen
