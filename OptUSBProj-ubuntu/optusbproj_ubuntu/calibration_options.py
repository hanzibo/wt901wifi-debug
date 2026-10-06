"""Shared, local-only wearing fit options; capture and analysis use snapshots."""
import math

DEFAULT_PRIOR_WEIGHT = 0.15


def validate_prior_weight(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError('穿戴标定先验权重 λ 须为 0～2 的有限数值')
    return value


def capture_prior_weight(guide, override=None):
    return validate_prior_weight(guide.get('wearing_prior_weight', DEFAULT_PRIOR_WEIGHT)
                                 if override is None else override)


def add_prior_control(dialog, owner, *, visual=False):
    from PySide6 import QtWidgets
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(0, 2); spin.setDecimals(3); spin.setSingleStep(.05)
    spin.setValue(getattr(owner, 'wearing_prior_weight', DEFAULT_PRIOR_WEIGHT))
    spin.setToolTip('0：不使用手动映射弱先验；默认0.15。仅影响后续拟合，不改变已有映射。')
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel('穿戴先验权重 λ' if not visual else '原IMU对照先验权重 λ'))
    row.addWidget(spin)
    spin.prior_buttons = []
    for label, value in (('禁用弱先验', 0), ('恢复默认', DEFAULT_PRIOR_WEIGHT)):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(lambda checked=False, v=value: spin.setValue(v) if spin.isEnabled() else None)
        row.addWidget(button)
        spin.prior_buttons.append(button)
    hint = QtWidgets.QLabel()
    hint.setWordWrap(True)
    def update_hint(value):
        hint.setText(('视觉主拟合不使用此先验；仅用于原IMU对照。' if visual else '') +
                     ('λ=0：关闭手动映射弱先验；动作先验与验收条件不变。' if value == 0 else
                      f'λ={value:.3f}；仅影响后续拟合。' + ('注意：大于0.3，可能不再是弱先验。' if value > .3 else '')))
    spin.update_prior_hint = update_hint
    def changed(value):
        update_hint(value)
        if hasattr(owner, 'set_wearing_prior_weight'):
            try: owner.set_wearing_prior_weight(value)
            except (OSError, ValueError) as exc:
                spin.blockSignals(True); spin.setValue(owner.wearing_prior_weight); spin.blockSignals(False)
                update_hint(spin.value())
                QtWidgets.QMessageBox.warning(dialog, '参数未保存', str(exc))
    spin.valueChanged.connect(changed)
    # Initial display must not write configuration.
    update_hint(spin.value())
    dialog.layout().insertLayout(1, row)
    dialog.layout().insertWidget(2, hint)
    dialog.prior_weight_spin = spin
    dialog.prior_owner = owner
    dialog.prior_capture_label = QtWidgets.QLabel('本轮参数：尚未开始采集／计算')
    dialog.layout().insertWidget(3, dialog.prior_capture_label)
    if hasattr(owner, '_prior_spins'):
        owner._prior_spins.append(spin)
        spin.destroyed.connect(lambda: owner._prior_spins.remove(spin) if spin in owner._prior_spins else None)
    return spin


def freeze_prior_control(dialog):
    dialog.capture_prior_weight = dialog.prior_weight_spin.value()
    dialog.prior_capture_label.setText(f'本轮采集 λ={dialog.capture_prior_weight:.3f}（已冻结）')
    set_prior_enabled(dialog, False)


def set_prior_enabled(dialog, enabled):
    spin = dialog.prior_weight_spin
    spin.setEnabled(enabled)
    for button in spin.prior_buttons:button.setEnabled(enabled)
    if enabled:
        spin.blockSignals(True)
        spin.setValue(getattr(dialog.prior_owner, 'wearing_prior_weight', spin.value()))
        spin.blockSignals(False)
        spin.update_prior_hint(spin.value())
