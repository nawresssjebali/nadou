import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CourierDetailsComponent } from './courier-details.component';

describe('CourierDetailsComponent', () => {
  let component: CourierDetailsComponent;
  let fixture: ComponentFixture<CourierDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CourierDetailsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CourierDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
