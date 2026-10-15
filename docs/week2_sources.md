# Week 2 source registry and provenance

**Corpus snapshot:** 2026-10-15  
**Access date for web sources:** 2026-10-15

The knowledge base describes a fictional Vietnamese electronics retailer. It is a **synthetic-derived corpus**, not the policy set of any named company. Public sources are used to establish realistic rule types, eligibility checks, safety boundaries and operating patterns. Wording, identifiers and the final combination of rules are original to this project.

`provenance: synthetic-derived` means the document adapts public rules or manufacturer guidance. `provenance: operational-synthesis` means the public sources establish the customer promise and required checks, while the internal agent workflow is a documented design assumption because real internal SOPs are generally not public.

## Retail policy sources

### SRC-DMX-RETURNS

- Source: [Điện Máy Xanh - Chính sách bảo hành, đổi trả sản phẩm](https://www.dienmayxanh.com/chinh-sach-bao-hanh-san-pham)
- Authority: official retailer policy.
- Used for: condition checks, product-category exceptions, accessories/packaging checks, manufacturer-fault handling and the distinction between warranty and exchange.
- Adaptation: the project normalises the many product-specific tables into simpler fictional-retailer rules; it does not claim Điện Máy Xanh's exact periods or fees.

### SRC-FPT-EXCHANGE

- Source: [FPT Shop - Chính sách đổi sản phẩm](https://fptshop.com.vn/ho-tro/chinh-sach-doi-san-pham)
- Authority: official retailer policy.
- Used for: checking order evidence, physical condition, serial number, accessories and manufacturer warranty eligibility.
- Adaptation: cross-checked with `SRC-DMX-RETURNS`; category-specific provisions are simplified for controlled evaluation.

### SRC-FPT-DELIVERY

- Source: [FPT Shop - Chính sách giao hàng](https://fptshop.com.vn/ho-tro/chinh-sach-giao-hang)
- Authority: official retailer policy.
- Used for: delivery zones, delivery lead-time ranges, distance-based fees, installation distinctions and handover responsibilities.
- Adaptation: the fictional thresholds are kept internally consistent and intentionally easier to label than the real location/product matrix.

### SRC-FPT-TOS

- Source: [FPT Shop - Điều khoản sử dụng](https://fptshop.com.vn/ho-tro/tos)
- Authority: official retailer terms.
- Used for: ordering, payment, identity/contact information, transaction errors and customer responsibilities.
- Adaptation: transformed into short agent-facing rules and FAQs; no contractual wording is copied.

### SRC-FPT-INSTALLMENT

- Source: [FPT Shop - Mua trả góp](https://fptshop.com.vn/ho-tro/mua-tra-gop)
- Authority: official retailer service information.
- Used for: the separation between card instalments and finance-company instalments, eligibility dependence on providers and the need to disclose fees/terms.
- Adaptation: exact banks, promotional rates and periodically changing campaigns are omitted from the stable corpus.

## Legal and regulatory sources

### SRC-DECREE-13

- Source: [Government of Vietnam - Decree 13/2023/ND-CP on personal-data protection](https://vanban.chinhphu.vn/?pageid=27160&docid=207759)
- Authority: official legal text.
- Used for: purpose limitation, appropriate handling of personal data and controlled disclosure.
- Adaptation: translated into minimum identity-verification steps for the fictional support desk; this is a dataset rule, not legal advice.

### SRC-CONSUMER-LAW

- Source: [Ministry of Industry and Trade - notable points of the 2023 Law on Protection of Consumer Rights](https://moit.gov.vn/tin-tuc/bao-chi-voi-nguoi-dan/luat-bao-ve-quyen-loi-nguoi-tieu-dung-nam-2023-nhung-diem-moi-noi-bat.html)
- Authority: official ministry explanation.
- Used for: complaint channels, transparent information, receipt of consumer requests and escalation of unresolved cases.
- Adaptation: internal response targets are explicit fictional service-level objectives, not statutory deadlines.

### SRC-EINVOICE-123

- Source: [Government of Vietnam - Decree 123/2020/ND-CP on invoices and documents](https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-123-2020-nd-cp-quy-dinh-ve-hoa-don-chung-tu-119230809110550559.htm)
- Authority: official government publication.
- Used for: required invoice information and the distinction between issuance and correction.
- Adaptation: the support workflow and two-day service target are fictional operating rules.

### SRC-SBP-OTP

- Source: [State Bank of Vietnam - customer warning on protecting banking information and OTP codes](https://www.sbv.gov.vn/)
- Authority: official regulator website; the registry points to the publisher because individual warnings are periodically reorganised.
- Used for: never requesting PIN, CVV, passwords or OTP and escalating suspected compromise to the customer's bank.
- Adaptation: expressed as a mandatory customer-support safety rule.

## Manufacturer and platform support sources

### SRC-SAMSUNG-WARRANTY

- Source: [Samsung Vietnam - Warranty information](https://www.samsung.com/vn/support/warranty/)
- Authority: official manufacturer support.
- Used for: serial/IMEI evidence, warranty exclusions, authorised-service handling and manufacturer-dependent warranty periods.

### SRC-SAMSUNG-SUPPORT

- Source: [Samsung Vietnam - Product support](https://www.samsung.com/vn/support/)
- Authority: official manufacturer support portal.
- Used for: safe first-line checks and the boundary between customer checks and authorised service.

### SRC-SAMSUNG-WASHER

- Source: [Samsung Vietnam - Washing-machine support](https://www.samsung.com/vn/support/home-appliances/)
- Authority: official manufacturer support portal.
- Used for: checking water supply/drain paths, levelling, filters, door locks and disconnecting power before maintenance.

### SRC-SAMSUNG-FRIDGE

- Source: [Samsung Vietnam - Refrigerator support](https://www.samsung.com/vn/support/home-appliances/)
- Authority: official manufacturer support portal.
- Used for: temperature, door seal, airflow, installation clearance, noises and safe defrosting.

### SRC-SAMSUNG-TV

- Source: [Samsung Vietnam - TV support](https://www.samsung.com/vn/support/tv-audio-video/)
- Authority: official manufacturer support portal.
- Used for: power reset, input isolation, network checks and remote-control checks.

### SRC-LG-SUPPORT

- Source: [LG Vietnam - Product support](https://www.lg.com/vn/tro-giup/)
- Authority: official manufacturer support portal.
- Used for: cross-checking appliance troubleshooting patterns, cleaning/installation checks and conditions requiring service.

### SRC-APPLE-LIQUID

- Source: [Apple Support - Water, splash and dust resistance of iPhone](https://support.apple.com/vi-vn/108039)
- Authority: official manufacturer support.
- Used for: water resistance is not permanent, avoiding charging a wet device and treating liquid damage separately from normal warranty.

### SRC-APPLE-RESTART

- Source: [Apple Support - Force restart iPhone](https://support.apple.com/vi-vn/guide/iphone/iph8903c3ee6/ios)
- Authority: official manufacturer support.
- Used for: safe force-restart guidance without presenting it as a data-erasing reset.

### SRC-MICROSOFT-WIFI

- Source: [Microsoft Support - Fix Wi-Fi connection issues in Windows](https://support.microsoft.com/vi-vn/windows/khắc-phục-sự-cố-kết-nối-wi-fi-trong-windows-9424a1f7-6a3b-65a6-4d78-7f07eee84d2c)
- Authority: official platform support.
- Used for: restart, airplane-mode, forget/reconnect, router isolation and driver checks.

### SRC-MICROSOFT-BATTERY

- Source: [Microsoft Support - Caring for your battery in Windows](https://support.microsoft.com/vi-vn/windows/chăm-sóc-pin-của-bạn-trong-windows-2db3e37f-5e7d-488e-9086-ed15320519e4)
- Authority: official platform support.
- Used for: heat awareness, battery-health precautions and stopping use for swelling or abnormal heat.

## Evidence limitations

- Web policies change. The access date and corpus snapshot make this dataset reproducible, but sources should be archived or rechecked before the final report.
- Several retailer policies vary by product category, brand, sales channel and promotion. The fictional corpus deliberately consolidates them into a smaller consistent rule set suitable for labelled experiments.
- Internal procedures are rarely public. Their action order, ownership and service-level targets are therefore explicit research assumptions informed by public customer promises, not claims about a real company's private workflow.
- Troubleshooting instructions are generalised across brands. A production system would retrieve model-specific manuals and must not replace authorised technical service.

## Controlled assumptions requiring project approval

The sources establish that these rule dimensions exist, but they do not justify pretending that one real retailer uses this exact combined set. The following values are explicit benchmark-design assumptions for the fictional company:

| Assumption | Value in corpus | Why retained |
|---|---|---|
| Warranty handling SLA | 15 working days; a loan device may be available after that point | Creates a precise normal-ticket fact and a multi-condition retrieval case |
| Manufacturer-fault exchange | 30 days | A normalised window across category-dependent retailer policies |
| Change-of-mind exchange | 7 days and 10% fee | Creates a clear distinction from manufacturer fault; must not be attributed to a referenced retailer |
| Refund processing | 5-10 working days | Represents payment-provider processing latency and supports timeline questions |
| Fictional delivery benefit | Free from VND 500,000 within 20 km | Stable benchmark threshold; real retailers use different matrices |
| COD ceiling | VND 20,000,000 | Fictional fraud/risk-control threshold |
| Card-instalment floor | VND 3,000,000 | Stable eligibility threshold; real campaigns vary by bank and date |
| Complaint SLA | acknowledge in 1 working day, initial response in 3 | Fictional internal service objective, not a statutory deadline |
| E-invoice delivery SLA | 2 working days | Fictional internal service objective, separate from legal invoicing duties |

These assumptions are acceptable only because the corpus is explicitly fictional and internally consistent. They should be reviewed by the supervisor/domain reviewer and frozen before ticket labelling. The final report must cite this table whenever it describes dataset realism; it must not say that every numeric threshold was copied from a real company.
