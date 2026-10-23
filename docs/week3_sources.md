# Week 3 source registry and provenance

**Corpus snapshot:** 2026-10-22  
**Access date for web sources:** 2026-10-22

Week 3 adds two kinds of evidence to the Week-2 registry ([week2_sources.md](week2_sources.md)):

- `provenance: source-extracted`: product catalogues (`SP-`) and manufacturer warranty documents (`NSX-`). Every number, model code, capacity, port count, warranty period and price in these documents was read from the official page listed below on the access date and translated into Vietnamese. Fields a page does not publish are written as "chưa có dữ liệu" (no data) instead of being estimated. Only manufacturer sites, the manufacturers' Vietnamese newsrooms and Lenovo PSREF were used; no retailer or review site was used for product facts.
- `provenance: controlled-variant`: the expired policy and the six documents that carry planted conflicts. They are deliberate variants of a Week-2 rule; their topic sources are the Week-2 retail sources, and the changed values are design assumptions recorded in `data/eval/kb_conflicts.json`.

The fictional retailer is not presented as an authorised dealer of any brand. Product facts are real; the store rules that wrap them (how an agent uses a catalogue, which data are looked up live) are `operational-synthesis`.

## Apple

### SRC-APPLE-IP18PRO

- Source: [iPhone 18 Pro - Tech Specs (Apple Support, IN)](https://support.apple.com/en-in/148590)
- Used for: capacities, display, A20 Pro chip, cameras, battery, charging, SIM, IP rating, size, weight, colours, box contents.

### SRC-APPLE-IPAIR

- Source: [iPhone Air - Tech Specs (Apple Support, HK)](https://support.apple.com/en-hk/125092)
- Used for: all iPhone Air facts, including dual-eSIM only on the Hong Kong page.

### SRC-APPLE-IP17

- Source: [iPhone 17 - Tech Specs (Apple Support, SG)](https://support.apple.com/en-sg/125089); cross-checked with the [US page](https://support.apple.com/en-us/125089)
- Used for: all iPhone 17 facts; the SIM difference between the SG page (nano-SIM + eSIM) and the US page (eSIM only).

### SRC-APPLE-IP17E

- Source: [iPhone 17e - Tech Specs (Apple Support, SG)](https://support.apple.com/en-sg/126470)
- Used for: all iPhone 17e facts.

### SRC-APPLE-IP16

- Source: [iPhone 16 - Tech Specs (Apple Support, SG)](https://support.apple.com/en-sg/121029)
- Used for: all iPhone 16 facts.

### SRC-APPLE-MBA13-M5

- Source: [MacBook Air (13-inch, M5) - Tech Specs (Apple Support, IN)](https://support.apple.com/en-in/126320)
- Used for: chip, memory and SSD options, display, battery, ports, wireless, weight, colours, box contents and adapter note.

### SRC-APPLE-MBA15-M5

- Source: [MacBook Air (15-inch, M5) - Tech Specs (Apple Support, IN)](https://support.apple.com/en-in/126321)
- Used for: all MacBook Air 15-inch M5 facts.

### SRC-APPLE-MBP14-M5

- Source: [MacBook Pro (14-inch, M5) - Tech Specs (Apple Support, SG)](https://support.apple.com/en-sg/125405)
- Used for: all MacBook Pro 14-inch M5 facts.

### SRC-APPLE-WARRANTY-IOS-VN

- Source: [Apple - iOS warranty, Rest of APAC, Vietnamese](https://www.apple.com/legal/warranty/products/ios-warranty-rest-of-apac-vietnamese.html)
- Used for: one-year limited warranty from the original retail purchase, exclusions, proof of purchase, data backup.

### SRC-APPLE-WARRANTY-MAC-VN

- Source: [Apple - Mac warranty, Rest of APAC, Vietnamese](https://www.apple.com/legal/warranty/products/embedded-mac-warranty-rest-of-apac-vietnamese.html)
- Used for: one-year Mac warranty, battery as a consumable, normal wear excluded.

## Samsung

`SRC-SAMSUNG-WARRANTY` (Samsung Vietnam warranty information) is already registered in Week 2 and is reused by `NSX-SAMSUNG-2026` for the warranty-period table, start-date rule, exclusions and replacement-part rules.

### SRC-SAMSUNG-S26-VN

- Source: [Samsung Newsroom Vietnam - Galaxy S26 series launch](https://news.samsung.com/vn/samsung-ra-mat-galaxy-s26-series-galaxy-ai-phone-truc-quan-nhat-tu-truoc-den-nay)
- Used for: Galaxy S26, S26+ and S26 Ultra specifications, Vietnamese recommended retail prices at launch, colours, pre-order dates.

### SRC-SAMSUNG-A57-A37-VN

- Source: [Samsung Newsroom Vietnam - Galaxy A57 5G and A37 5G launch](https://news.samsung.com/vn/samsung-ra-mat-galaxy-a57-5g-va-galaxy-a37-5g-trai-nghiem-cac-tinh-nang-chuyen-nghiep-voi-muc-gia-hap-dan)
- Used for: specifications, recommended retail prices, 20/03/2026 sale date, update policy.

### SRC-SAMSUNG-TV2026-VN

- Source: [Samsung Newsroom Vietnam - Samsung AI TV 2026 in Vietnam](https://news.samsung.com/vn/samsung-ai-tv-2026-da-co-mat-tai-viet-nam-mo-rong-danh-muc-ai-tv-voi-tro-ly-vision-ai-da-nhiem)
- Used for: series starting prices and the 01/04-30/06/2026 promotion (recorded as ended).

### SRC-SAMSUNG-QN80H

- Source: [Samsung Vietnam - 55" Neo QLED QN80H QA55QN80HAKXXV](https://www.samsung.com/vn/tvs/qled-tv/qn80h-55-inch-neo-qled-4k-mini-led-smart-tv-qa55qn80hakxxv/)

### SRC-SAMSUNG-QN70F

- Source: [Samsung Vietnam - 55" Neo QLED QN70F QA55QN70FAKXXV](https://www.samsung.com/vn/tvs/qled-tv/qn70f-55-inch-neo-qled-4k-vision-ai-smart-tv-qa55qn70fakxxv/)

### SRC-SAMSUNG-U8000H

- Source: [Samsung Vietnam - 43" Crystal UHD U8000H UA43U8000HKXXV](https://www.samsung.com/vn/tvs/uhd-4k-tv/u8000h-43-inch-crystal-uhd-4k-smart-tv-ua43u8000hkxxv/)

### SRC-SAMSUNG-S90F

- Source: [Samsung Vietnam - 65" OLED S90F QA65S90FAEXXV](https://www.samsung.com/vn/tvs/oled-tv/s90f-65-inch-oled-4k-smart-tv-qa65s90faexxv/)

### SRC-SAMSUNG-RT31CB

- Source: [Samsung Vietnam - Bespoke AI top-mount freezer 305 L RT31CB56248ASV](https://www.samsung.com/vn/refrigerators/top-mount-freezer/rt5300c-top-mount-freezer-bespoke-305l-rt31cb56248asv/)

### SRC-SAMSUNG-WW10DB

- Source: [Samsung Vietnam - Bespoke AI front-load washer 10 kg WW10DB7U34GBSV](https://www.samsung.com/vn/washers-and-dryers/washing-machines/ww7000d-front-loading-smartthings-ai-energy-made-a-20-percent-extra-energy-efficiency-ai-ecobubble-10kg-black-ww10db7u34gbsv/)
- Used for: specifications and the 10-year Digital Inverter motor warranty for purchases from 01/04/2026.

### SRC-SAMSUNG-MG23T

- Source: [Samsung Vietnam - Grill microwave 23 L MG23T5018CK](https://www.samsung.com/vn/microwave-ovens/grill/microwaves-oven-grill-mg23t5018ck/)

### SRC-SAMSUNG-MS23K

- Source: [Samsung Vietnam - Solo microwave 23 L MS23K3513AS](https://www.samsung.com/vn/microwave-ovens/solo/microwaves-oven-solo-ms23k3513as/)

## LG

### SRC-LG-WARRANTY-VN

- Source: [LG Vietnam - Warranty policy](https://www.lg.com/vn/tro-giup/bao-hanh/); hotline from [LG Vietnam - Product support warranty page](https://www.lg.com/vn/tro-giup/ho-tro-san-pham/bao-hanh/)
- Used for: periods by category for purchases from 24/02/2023, caps from the manufacturing date, start-date rules, documents, exclusions. `SRC-LG-SUPPORT` (Week 2) is cited for the support channel.

### SRC-LG-OLED55C5

- Source: [LG Vietnam - OLED evo AI C5 55" OLED55C5PSA](https://lg.com/vn/tv-va-loa-thanh/tv-oled/oled55c5psa)

### SRC-LG-55QNED70

- Source: [LG Vietnam - QNED70 55" 55QNED70ASA](https://www.lg.com/vn/tv-va-loa-thanh/tv-qned/55qned70asa/)

### SRC-LG-55QNED86

- Source: [LG Vietnam - QNED evo QNED86 55" 55QNED86ASA](https://www.lg.com/vn/tv-va-loa-thanh/qned-evo/55qned86asa/)

### SRC-LG-LTD33BLM

- Source: [LG Vietnam - Top-freezer Smart Inverter 332 L LTD33BLM](https://www.lg.com/vn/tu-lanh/tu-lanh-ngan-da-tren/ltd33blm/)

### SRC-LG-FV1410S4B

- Source: [LG Vietnam - AI DD front-load washer 10 kg FV1410S4B](https://www.lg.com/vn/giat-say/may-giat-long-ngang/fv1410s4b/)

### SRC-LG-V10API1

- Source: [LG Vietnam - DUALCOOL Inverter 9,000 BTU V10API1](https://www.lg.com/vn/dieu-hoa/v10api1/)
- Note: the specification table on the page uses unit codes S3-Q09JA2P5 / S3UQ09JA2P5; the catalogue records both names.

## Lenovo

### SRC-LENOVO-E14G7

- Source: [Lenovo PSREF - ThinkPad E14 Gen 7 (Intel) platform specifications (PDF)](https://psref.lenovo.com/syspool/Sys/PDF/ThinkPad/ThinkPad_E14_Gen_7_Intel/ThinkPad_E14_Gen_7_Intel_Spec.pdf); Vietnamese model listed at [PSREF 21SX0032VN](https://psref.lenovo.com/Detail/ThinkPad_E14_Gen_7_Intel?M=21SX0032VN)
- Used for: platform options and base-warranty options. The per-model detail page renders with JavaScript and could not be read, so the catalogue describes the platform and requires the machine-type model for configuration questions.

### SRC-LENOVO-SLIM5-14IRH10

- Source: [Lenovo PSREF - IdeaPad Slim 5 14IRH10 platform specifications (PDF)](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_Slim_5_14IRH10/IdeaPad_Slim_5_14IRH10_Spec.pdf); Vietnamese model listed at [PSREF 83HR0000VN](https://psref.lenovo.com/Detail/IdeaPad_Slim_5_14IRH10?M=83HR0000VN)

## Daikin

### SRC-DAIKIN-WARRANTY-VN

- Source: [Daikin Vietnam - Warranty policy](https://www.daikin.com.vn/dich-vu/dich-vu-dan-dung/chinh-sach-bao-hanh)
- Used for: 12 months from installation, 5-year compressor, circuit-board and Microchannel terms, start-date order, activation channels, exclusions, hotline.

### SRC-DAIKIN-FTKF25

- Source: [Daikin E-Shop - FTKF25ZVMV 1HP](https://www.daikin.com.vn/shop/san-pham/loai-2-dan-roi/dieu-hoa-inverter-ftkf25zvmv-1hp)
- Used for: specifications and the listed and special prices on the access date.

### SRC-DAIKIN-FTKY25

- Source: [Daikin E-Shop - FTKY25ZVMV 1HP](https://www.daikin.com.vn/shop/san-pham/loai-2-dan-roi/dieu-hoa-inverter-ftky25zvmv-1hp)
- Used for: specifications, prices on the access date and the Premier Care offer of the e-shop.

## Panasonic

### SRC-PANASONIC-WARRANTY-VN

- Source: [Panasonic Vietnam - Product warranty periods](https://www.panasonic.com/vn/support/product-warranty-period.html)
- Used for: the policy valid 28/07/2026-31/03/2027, periods by product group and origin, special-part periods, start-date rules. The document's `effective_date` and `expiry_date` copy this validity window.

### SRC-PANASONIC-NRBV361

- Source: [Panasonic Vietnam - 2-door bottom-freezer NR-BV361GPKV](https://www.panasonic.com/vn/consumer/home-appliances/fridge/2-door-bottom-freezer/nr-bv361gpkv.html)

## Philips

### SRC-PHILIPS-NA332

- Source: [Philips Vietnam - Airfryer 3000 Series 6.2 L NA332/00](https://www.philips.com.vn/c-p/NA332_00/3000-series-airfryer-6.2l)

### SRC-PHILIPS-NA130

- Source: [Philips Vietnam - Airfryer 1000 Series 6.2 L NA130/00](https://www.philips.com.vn/c-p/NA130_00/1000-series-airfryer-1000-series-62l)

### SRC-PHILIPS-HD9257

- Source: [Philips Vietnam - Airfryer 3000 Series XL HD9257/80](https://www.philips.com.vn/c-p/HD9257_80/airfryer-3000-series-xl-digital-window)

## Simulated operating documents

Real retailers do not publish their internal fee tables, membership rules, extended-warranty terms or reply standards in full. Thirteen documents therefore use `provenance: simulated`: the values are chosen by the project, calibrated against the public references below, and constrained by the consistency rules in `SRC-SIM-RULES`. They are not attributed to any real company.

### SRC-SIM-RULES

- Source: simulation rules of this project (listed under "Used for" below), summarised in [week3.md](week3.md).
- Authority: design document of this benchmark, not an external source.
- Used for: the rules every simulated document must satisfy: (1) no value may contradict an active Week-2 or Week-3 document except where registered in `data/eval/kb_conflicts.json`; (2) simulated documents refer to existing policies instead of restating their numbers (refund days, repair days, free-delivery threshold); (3) magnitudes follow a dated public reference when one exists; (4) topics kept as `no_evidence` in Week 3 (rental, trade-in, live prices) stay unanswered.

### SRC-META-AC-INSTALL

- Source: [META.vn - Bảng giá phí và phụ kiện lắp đặt điều hòa](https://meta.vn/hotro/bang-gia-phi-va-phu-kien-lap-dat-dieu-hoa-445) (article dated 07/02/2026)
- Authority: public retailer price list.
- Used for: calibrating copper pipe (VND 200,000-260,000/m by capacity), outdoor brackets (VND 120,000-280,000), breaker (VND 120,000), drain pipe (VND 10,000-40,000/m), uninstall labour (VND 200,000), core drilling (VND 100,000-200,000). The simulated price list uses the same order of magnitude, not META's table.

### SRC-HACOM-EXT-WARRANTY

- Source: [HACOM - Gói bảo hành 1 năm nhà sản xuất + 1 năm BHMR cho laptop 35-40 triệu](https://hacom.vn/goi-bao-hanh-1-nam-nha-san-xuat-1-nam-bhmr-cua-laptop-co-muc-gia-ban-35-trieu-den-40-trieu)
- Authority: public retailer product page.
- Used for: calibrating the extended-warranty fee (VND 1,494,999 list price for a 1-year extension of a VND 35-40 million laptop, about 4%) and benefit structure (parts and labour, unlimited repairs, cap at product value, replacement when unrepairable).

### SRC-META-DEPOSIT

- Source: [META.vn - Quy định về đặt cọc và giữ hàng](https://meta.vn/quy-dinh-ve-dat-coc-va-giu-hang)
- Authority: public retailer policy.
- Used for: the structure of deposit rules (deposit percentage, holding period, refund when the store cannot supply). The simulated policy chooses a customer-friendlier variant (10% deposit, full refund on cancellation) and states it as a project choice.

### SRC-SAMSUNG-REWARDS

- Source: [Samsung Vietnam - Giá trị điểm thưởng trong Samsung Rewards](https://www.samsung.com/vn/support/apps-services/how-much-is-one-rewards-point-worth)
- Authority: official manufacturer support page.
- Used for: evidence that point programmes fix a VND value per point and reserve the right to change it; the simulated membership policy uses its own value (1 point = VND 1,000, earned per VND 100,000).

## Evidence limitations

- **Regional pages.** Apple publishes Tech Specs per region; Vietnam-specific pages were not available for every model, so Singapore, Hong Kong and India pages were used and the region is stated wherever it matters (SIM type, power adapter in the box).
- **Missing fields are left missing.** Several official pages omit values (for example LG OLED C5 speaker power, Philips airfryer power and dimensions, Panasonic energy label). The catalogue marks them as unavailable. These gaps are intentional `no_evidence` material for Week 4.
- **Prices change.** Samsung prices are launch recommended retail prices from press releases; Daikin prices are e-shop prices on the access date. Both are labelled as such, and the store procedure says current prices come from the sales system.
- **Sources can disagree with themselves.** The Samsung Vietnam warranty page sets the start date at first valid start-up in its table but refers to the delivery date in its policy text; the KB records only the table rule and this note keeps the discrepancy visible. LG's spec table for V10API1 uses different unit codes from the page title.
- **Snapshot.** Pages were read through a text extractor on 2026-10-22. Before the final report the pages should be archived (e.g. saved as PDF) because product pages are updated without version history.
